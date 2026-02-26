"""
run_cashflow_comparison.py — 두 자본구조 시나리오 현금흐름표 비교

1) 선순위대출 80% + 지분투자 20% (전통 PF)
2) 토큰 40% + 선순위대출 40% + 지분투자 20% (토큰화 PF)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 공통 파라미터
# ============================================================
CAPITAL_TOTAL = 2_000_000_000        # 총 사업비 20억
EQUITY_AMOUNT = 400_000_000          # 지분투자 4억 (고정)
SENIOR_INTEREST_RATE = 0.06          # 선순위 이자율 6%
DEBT_REPAYMENT_YEARS = 15            # 대출 상환기간 15년
YEARS = 20                           # 운영기간 20년

# 비용
OPEX_ANNUAL = 10_000_000
ADMIN_FEE_ANNUAL = 10_000_000
MAJOR_REPAIR_ANNUAL = 5_000_000
OPEX_TOTAL = OPEX_ANNUAL + ADMIN_FEE_ANNUAL + MAJOR_REPAIR_ANNUAL

# 발전 및 가격
SMP_PRICE = 113.0                    # 원/kWh
REC_PRICE = 77.0                     # 원/kWh
MONTHLY_GEN = 105.9                  # MWh/월
ANNUAL_GEN_BASE = MONTHLY_GEN * 12   # 1,270.8 MWh/년
DEGRADATION_RATE = 0.005             # 연 0.5% 열화

UNIT = 1e6  # 표시 단위: 백만원

# ============================================================
# 시나리오 1: 선순위 80% + 지분 20% (전통 PF)
# ============================================================
SCENARIO_1 = {
    'name': '전통 PF (대출 80%)',
    'senior_debt': 1_600_000_000,     # 선순위 16억 (80%)
    'token_amount': 0,                 # 토큰 없음
    'equity_amount': 400_000_000,      # 지분 4억 (20%)
    'token_share': 0.0,                # 토큰 귀속 비율
    'equity_share': 1.0,               # 지분 귀속 비율 (전액)
}

# ============================================================
# 시나리오 2: 토큰 40% + 선순위 40% + 지분 20% (토큰화 PF)
# ============================================================
SCENARIO_2 = {
    'name': '토큰화 PF (토큰 40%)',
    'senior_debt': 800_000_000,        # 선순위 8억 (40%)
    'token_amount': 800_000_000,       # 토큰 8억 (40%)
    'equity_amount': 400_000_000,      # 지분 4억 (20%)
    'token_share': 2/3,                # 토큰 귀속 비율
    'equity_share': 1/3,               # 지분 귀속 비율
}


def calc_irr(cashflows, lo=-0.99, hi=2.0, tol=1e-10, max_iter=2000):
    """IRR 계산 (이분법)."""
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        npv = sum(cf / (1 + mid) ** t for t, cf in enumerate(cashflows))
        if abs(npv) < tol:
            return mid
        if npv > 0:
            lo = mid
        else:
            hi = mid
    return mid


def calc_scenario(scenario):
    """시나리오별 연도별 현금흐름 계산."""
    senior_debt = scenario['senior_debt']
    token_amount = scenario['token_amount']
    equity_amount = scenario['equity_amount']
    token_share = scenario['token_share']
    equity_share = scenario['equity_share']

    # 원리금균등상환 월 납입액
    r_m = SENIOR_INTEREST_RATE / 12
    n_months = DEBT_REPAYMENT_YEARS * 12
    if senior_debt > 0:
        monthly_pmt = senior_debt * r_m * (1 + r_m) ** n_months / ((1 + r_m) ** n_months - 1)
    else:
        monthly_pmt = 0

    data = {}
    token_cf_list = []
    equity_cf_list = []
    for y in range(1, YEARS + 1):
        d = {}

        # 발전량 (열화 반영)
        d['gen'] = ANNUAL_GEN_BASE * (1 - DEGRADATION_RATE) ** (y - 1)

        # 수익
        d['smp_rev'] = d['gen'] * SMP_PRICE * 1000
        d['rec_rev'] = d['gen'] * REC_PRICE * 1000
        d['revenue'] = d['smp_rev'] + d['rec_rev']

        # 비용
        d['opex_ops'] = float(OPEX_ANNUAL)
        d['opex_admin'] = float(ADMIN_FEE_ANNUAL)
        d['opex_repair'] = float(MAJOR_REPAIR_ANNUAL)
        d['opex_total'] = OPEX_TOTAL

        # NOI
        d['noi'] = d['revenue'] - d['opex_total']

        # 대출 상환 (원리금균등상환)
        if y <= DEBT_REPAYMENT_YEARS and senior_debt > 0:
            annual_int = 0.0
            annual_prin = 0.0
            for m in range(12):
                month_idx = (y - 1) * 12 + m
                if month_idx == 0:
                    remaining = senior_debt
                else:
                    remaining = senior_debt * ((1 + r_m) ** n_months - (1 + r_m) ** month_idx) / ((1 + r_m) ** n_months - 1)
                int_m = remaining * r_m
                prin_m = monthly_pmt - int_m
                annual_int += int_m
                annual_prin += prin_m
            d['interest'] = annual_int
            d['principal'] = annual_prin
        else:
            d['interest'] = 0.0
            d['principal'] = 0.0

        d['ds'] = d['interest'] + d['principal']
        d['dscr'] = d['noi'] / d['ds'] if d['ds'] > 0 else None

        # 잔여현금흐름
        d['cf_residual'] = d['noi'] - d['ds']

        # 배분
        d['token_cf'] = d['cf_residual'] * token_share
        d['equity_cf'] = d['cf_residual'] * equity_share

        # 현금흐름 리스트에 추가
        token_cf_list.append(d['token_cf'])
        equity_cf_list.append(d['equity_cf'])

        # IRR 계산 (누적)
        if token_amount > 0 and y >= 1:
            token_cfs = [-token_amount] + token_cf_list
            d['token_irr'] = calc_irr(token_cfs) * 100
        else:
            d['token_irr'] = None

        equity_cfs = [-equity_amount] + equity_cf_list
        d['equity_irr'] = calc_irr(equity_cfs) * 100

        data[y] = d

    return data


# ============================================================
# 테이블 행 정의
# ============================================================
def get_row_defs_scenario1():
    """시나리오 1 (전통 PF) 행 정의."""
    return [
        ('영업수익',           None, 'section'),
        ('  발전량 (MWh)',     'gen', 'mwh'),
        ('  SMP 수익',         'smp_rev', 'money'),
        ('  REC 수익',         'rec_rev', 'money'),
        ('  매출액',           'revenue', 'subtotal'),
        ('', None, 'blank'),
        ('영업비용',           None, 'section'),
        ('  운영비',           'opex_ops', 'money'),
        ('  사무위탁비',       'opex_admin', 'money'),
        ('  대수선비',         'opex_repair', 'money'),
        ('  OPEX 합계',        'opex_total', 'subtotal'),
        ('', None, 'blank'),
        ('  NOI (영업이익)',    'noi', 'subtotal'),
        ('', None, 'blank'),
        ('Debt Service',       None, 'section'),
        ('  이자상환',         'interest', 'money'),
        ('  원금상환',         'principal', 'money'),
        ('  DS (이자+원금)',    'ds', 'subtotal'),
        ('  DSCR',             'dscr', 'dscr'),
        ('', None, 'blank'),
        ('잔여현금 배분',      None, 'section'),
        ('  잔여현금흐름',     'cf_residual', 'subtotal'),
        ('  지분투자자 귀속',  'equity_cf', 'money'),
        ('', None, 'blank'),
        ('IRR (누적)',         None, 'section'),
        ('  지분투자 IRR',     'equity_irr', 'pct'),
    ]


def get_row_defs_scenario2():
    """시나리오 2 (토큰화 PF) 행 정의."""
    return [
        ('영업수익',           None, 'section'),
        ('  발전량 (MWh)',     'gen', 'mwh'),
        ('  SMP 수익',         'smp_rev', 'money'),
        ('  REC 수익',         'rec_rev', 'money'),
        ('  매출액',           'revenue', 'subtotal'),
        ('', None, 'blank'),
        ('영업비용',           None, 'section'),
        ('  운영비',           'opex_ops', 'money'),
        ('  사무위탁비',       'opex_admin', 'money'),
        ('  대수선비',         'opex_repair', 'money'),
        ('  OPEX 합계',        'opex_total', 'subtotal'),
        ('', None, 'blank'),
        ('  NOI (영업이익)',    'noi', 'subtotal'),
        ('', None, 'blank'),
        ('Debt Service',       None, 'section'),
        ('  이자상환',         'interest', 'money'),
        ('  원금상환',         'principal', 'money'),
        ('  DS (이자+원금)',    'ds', 'subtotal'),
        ('  DSCR',             'dscr', 'dscr'),
        ('', None, 'blank'),
        ('잔여현금 배분',      None, 'section'),
        ('  잔여현금흐름',     'cf_residual', 'subtotal'),
        ('  토큰 귀속 (2/3)',  'token_cf', 'money'),
        ('  지분투자자 귀속 (1/3)', 'equity_cf', 'money'),
        ('', None, 'blank'),
        ('IRR (누적)',         None, 'section'),
        ('  토큰 IRR',         'token_irr', 'pct'),
        ('  지분투자자 IRR',   'equity_irr', 'pct'),
    ]


def fmt_val(val, fmt_type):
    """값을 포맷 문자열로 변환."""
    if val is None:
        return '-'
    if fmt_type == 'mwh':
        return f'{val:,.1f}'
    if fmt_type in ('money', 'subtotal'):
        v = val / UNIT
        return f'{v:,.1f}' if abs(v) >= 0.05 else '-'
    if fmt_type == 'dscr':
        return f'{val:.2f}'
    if fmt_type == 'pct':
        return f'{val:.1f}%'
    return ''


# ============================================================
# 색상 정의
# ============================================================
C_HEADER   = '#2C3E50'
C_SECTION  = '#4472C4'
C_SUBTOTAL = '#D6E4F0'
C_BLANK    = '#F5F5F5'
C_DATA_ODD = '#FFFFFF'
C_DATA_EVEN = '#F2F6FA'


def build_and_render(data, row_defs, year_range, title, filename, subtitle=None):
    """주어진 연도 범위의 현금흐름표를 렌더링."""
    n_yr = len(year_range)
    col_labels = ['항목'] + [f'{y}년차' for y in year_range]
    n_cols = len(col_labels)

    cell_text = []
    row_formats = []

    for label, key, fmt in row_defs:
        if fmt == 'section':
            cell_text.append([label] + [''] * n_yr)
            row_formats.append('section')
            continue
        if fmt == 'blank':
            cell_text.append([''] * n_cols)
            row_formats.append('blank')
            continue

        row = [label]
        for y in year_range:
            d = data[y]
            val = d.get(key)
            row.append(fmt_val(val, fmt))
        cell_text.append(row)
        row_formats.append(fmt)

    n_rows = len(cell_text)

    fig, ax = plt.subplots(figsize=(18, 11))
    ax.axis('off')

    # 제목
    ax.set_title(title, fontsize=14, fontweight='bold', pad=35, loc='left')

    # 부제목 (자본구조) - 우측 상단
    if subtitle:
        ax.text(0.99, 1.06, subtitle, transform=ax.transAxes,
                fontsize=10, ha='right', va='bottom', color='#999999')

    # 단위 표시
    ax.text(0.99, 1.02, '(단위: 백만원)', transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom', color='#666666')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # 헤더 행
    for j in range(n_cols):
        table[0, j].set_facecolor(C_HEADER)
        table[0, j].set_text_props(color='white', fontweight='bold', fontsize=10)

    # 데이터 행
    data_row_idx = 0
    for i in range(n_rows):
        ri = i + 1
        fmt = row_formats[i]

        if fmt == 'section':
            for j in range(n_cols):
                table[ri, j].set_facecolor(C_SECTION)
                table[ri, j].set_text_props(color='white', fontweight='bold')
            table[ri, 0].set_text_props(ha='left')
        elif fmt == 'blank':
            for j in range(n_cols):
                table[ri, j].set_facecolor(C_BLANK)
                table[ri, j].set_height(0.008)
        elif fmt == 'subtotal':
            for j in range(n_cols):
                table[ri, j].set_facecolor(C_SUBTOTAL)
                table[ri, j].set_text_props(fontweight='bold')
            table[ri, 0].set_text_props(ha='left', fontweight='bold')
            data_row_idx += 1
        else:
            bg = C_DATA_ODD if data_row_idx % 2 == 0 else C_DATA_EVEN
            for j in range(n_cols):
                table[ri, j].set_facecolor(bg)
            table[ri, 0].set_text_props(ha='left')
            data_row_idx += 1

    # 첫 열 왼쪽 정렬, 나머지 오른쪽 정렬
    for i in range(n_rows + 1):
        table[i, 0].set_text_props(ha='left')
        for j in range(1, n_cols):
            if i > 0:
                table[i, j].set_text_props(ha='right')

    # 열 폭
    col_widths = [0.18] + [0.082] * n_yr
    for j, w in enumerate(col_widths):
        for i in range(n_rows + 1):
            table[i, j].set_width(w)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Saved: {filename}')
    plt.close()


# ============================================================
# 메인 실행
# ============================================================
if __name__ == '__main__':
    # 시나리오 계산
    data1 = calc_scenario(SCENARIO_1)
    data2 = calc_scenario(SCENARIO_2)

    # 시나리오 1: 전통 PF (대출 80%)
    row_defs1 = get_row_defs_scenario1()
    build_and_render(
        data1, row_defs1,
        list(range(1, 11)),
        '전통 PF 현금흐름표 (1년차~10년차)',
        'cashflow_traditional_1.png',
        '자본구조: 선순위대출 80% (16억) + 지분투자 20% (4억)'
    )
    build_and_render(
        data1, row_defs1,
        list(range(11, 21)),
        '전통 PF 현금흐름표 (11년차~20년차)',
        'cashflow_traditional_2.png',
        '자본구조: 선순위대출 80% (16억) + 지분투자 20% (4억)'
    )

    # 시나리오 2: 토큰화 PF (토큰 40%)
    row_defs2 = get_row_defs_scenario2()
    build_and_render(
        data2, row_defs2,
        list(range(1, 11)),
        '토큰화 PF 현금흐름표 (1년차~10년차)',
        'cashflow_tokenized_1.png',
        '자본구조: 토큰 40% (8억) + 선순위대출 40% (8억) + 지분투자 20% (4억)'
    )
    build_and_render(
        data2, row_defs2,
        list(range(11, 21)),
        '토큰화 PF 현금흐름표 (11년차~20년차)',
        'cashflow_tokenized_2.png',
        '자본구조: 토큰 40% (8억) + 선순위대출 40% (8억) + 지분투자 20% (4억)'
    )

    # 요약 출력
    print('\n=== 시나리오 1: 전통 PF (대출 80%) ===')
    print(f'5년차 지분 IRR: {data1[5]["equity_irr"]:.1f}%')
    print(f'10년차 지분 IRR: {data1[10]["equity_irr"]:.1f}%')
    print(f'15년차 지분 IRR: {data1[15]["equity_irr"]:.1f}%')
    print(f'20년차 지분 IRR: {data1[20]["equity_irr"]:.1f}%')
    total_equity_cf1 = sum(data1[y]['equity_cf'] for y in range(1, 21))
    print(f'20년 총 지분 배당: {total_equity_cf1/1e8:.1f}억원')

    print('\n=== 시나리오 2: 토큰화 PF (토큰 40%) ===')
    print(f'5년차 토큰 IRR: {data2[5]["token_irr"]:.1f}%')
    print(f'10년차 토큰 IRR: {data2[10]["token_irr"]:.1f}%')
    print(f'15년차 토큰 IRR: {data2[15]["token_irr"]:.1f}%')
    print(f'20년차 토큰 IRR: {data2[20]["token_irr"]:.1f}%')
    total_token_cf2 = sum(data2[y]['token_cf'] for y in range(1, 21))
    print(f'20년 총 토큰 배당: {total_token_cf2/1e8:.1f}억원')

    print(f'\n5년차 지분투자자 IRR: {data2[5]["equity_irr"]:.1f}%')
    print(f'20년차 지분투자자 IRR: {data2[20]["equity_irr"]:.1f}%')
    total_equity_cf2 = sum(data2[y]['equity_cf'] for y in range(1, 21))
    print(f'20년 총 지분투자자 배당: {total_equity_cf2/1e8:.1f}억원')
