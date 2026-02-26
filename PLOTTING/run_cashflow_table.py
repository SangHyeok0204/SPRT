"""
run_cashflow_table.py — 태양광 PF 연도별 현금흐름표 (이미지 포맷 재현)

행=항목, 열=연도 구조. 섹션별 구분.
수치 가정은 SolarPFModel.py에서 읽어옴.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from SolarPFModel import (
    SENIOR_DEBT, TOKEN_AMOUNT, EQUITY_AMOUNT, CAPITAL_TOTAL,
    SENIOR_INTEREST_RATE, DEBT_REPAYMENT_YEARS,
    OPEX_ANNUAL, ADMIN_FEE_ANNUAL, MAJOR_REPAIR_ANNUAL,
    T_OPERATION,
    SMP_PARAMS, REC_PARAMS, ENERGY_PARAMS,
    DEGRADATION_RATE_ANNUAL,
    R_DISCOUNT_ANNUAL,
)

# ============================================================
# 파라미터
# ============================================================
YEARS = T_OPERATION // 12                       # 20년
SMP_PRICE = SMP_PARAMS['initial_price']         # 113 원/kWh
REC_PRICE = REC_PARAMS['initial_price']         # 77 원/kWh
REC_WEIGHT = 1.0                                # REC 가중치 (모델과 동일)
MONTHLY_GEN = ENERGY_PARAMS['base_generation']  # 105.9 MWh/월
ANNUAL_GEN_BASE = MONTHLY_GEN * 12              # 1,270.8 MWh/년

# 원리금균등상환 (annuity) 월 납입액
_r_m = SENIOR_INTEREST_RATE / 12
_n_months = DEBT_REPAYMENT_YEARS * 12
MONTHLY_PMT = SENIOR_DEBT * _r_m * (1 + _r_m) ** _n_months / ((1 + _r_m) ** _n_months - 1)

# 현금흐름표 배분 비율 (1:1, 토큰 8억 : 지분투자자 8억)
CF_TOKEN_SHARE = 0.5
CF_EQUITY_SHARE = 0.5

UNIT = 1e6  # 표시 단위: 백만원

# ============================================================
# 연도별 계산 (법인세 제외)
# ============================================================
data = {}
for y in range(1, YEARS + 1):
    d = {}

    # 발전량 (열화 반영)
    d['gen'] = ANNUAL_GEN_BASE * (1 - DEGRADATION_RATE_ANNUAL) ** (y - 1)

    # 수익
    d['smp_rev'] = d['gen'] * SMP_PRICE * 1000   # 원
    d['rec_rev'] = d['gen'] * REC_PRICE * REC_WEIGHT * 1000
    d['revenue'] = d['smp_rev'] + d['rec_rev']

    # 비용
    d['opex_ops'] = float(OPEX_ANNUAL)
    d['opex_admin'] = float(ADMIN_FEE_ANNUAL)
    d['opex_repair'] = float(MAJOR_REPAIR_ANNUAL)
    d['opex_total'] = d['opex_ops'] + d['opex_admin'] + d['opex_repair']

    # NOI
    d['noi'] = d['revenue'] - d['opex_total']

    # 대출 상환 (원리금균등상환, 월별 계산 → 연간 합산)
    if y <= DEBT_REPAYMENT_YEARS:
        annual_int = 0.0
        annual_prin = 0.0
        for m in range(12):
            month_idx = (y - 1) * 12 + m
            # 잔액: 원리금균등상환 잔액 공식
            if month_idx == 0:
                remaining = SENIOR_DEBT
            else:
                remaining = SENIOR_DEBT * ((1 + _r_m) ** _n_months - (1 + _r_m) ** month_idx) / ((1 + _r_m) ** _n_months - 1)
            int_m = remaining * _r_m
            prin_m = MONTHLY_PMT - int_m
            annual_int += int_m
            annual_prin += prin_m
        d['interest'] = annual_int
        d['principal'] = annual_prin
    else:
        d['interest'] = 0.0
        d['principal'] = 0.0

    d['ds'] = d['interest'] + d['principal']
    d['dscr'] = d['noi'] / d['ds'] if d['ds'] > 0 else None

    # 잔여현금흐름 (법인세 없음)
    d['cf_residual'] = d['noi'] - d['ds']

    # 배분 (1:1)
    d['token_cf'] = d['cf_residual'] * CF_TOKEN_SHARE
    d['sponsor_cf'] = d['cf_residual'] * CF_EQUITY_SHARE

    data[y] = d

# 할인 누적수익률 계산: Σ(CF_y / (1+r)^y) / 초기투자 × 100
r = R_DISCOUNT_ANNUAL  # 4.5%
cum_pv_noi = 0.0
cum_pv_token = 0.0
for y in range(1, YEARS + 1):
    df = (1 + r) ** y  # 할인계수
    cum_pv_noi += data[y]['noi'] / df
    cum_pv_token += data[y]['token_cf'] / df
    data[y]['cum_return_project'] = cum_pv_noi / CAPITAL_TOTAL * 100
    data[y]['cum_return_token'] = cum_pv_token / TOKEN_AMOUNT * 100

# ============================================================
# IRR 계산
# ============================================================
def calc_irr(cashflows, lo=-0.5, hi=1.0, tol=1e-10, max_iter=2000):
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

years_list = list(range(1, YEARS + 1))
noi_arr = np.array([data[y]['noi'] for y in years_list])
token_arr = np.array([data[y]['token_cf'] for y in years_list])

project_irr = calc_irr(np.concatenate([[-CAPITAL_TOTAL], noi_arr]))
token_irr = calc_irr(np.concatenate([[-TOKEN_AMOUNT], token_arr]))

# ============================================================
# 테이블 행 정의 (법인세 제거, 단가 제거)
# ============================================================
ROW_DEFS = [
    # ── 영업수익 ──
    ('영업수익',           None, 'section'),
    ('  발전량 (MWh)',     'gen', 'mwh'),
    ('  SMP 수익',         'smp_rev', 'money'),
    ('  REC 수익',         'rec_rev', 'money'),
    ('  매출액',           'revenue', 'subtotal'),
    ('', None, 'blank'),
    # ── 영업비용 ──
    ('영업비용',           None, 'section'),
    ('  운영비',           'opex_ops', 'money'),
    ('  사무위탁비',       'opex_admin', 'money'),
    ('  대수선비',         'opex_repair', 'money'),
    ('  OPEX 합계',        'opex_total', 'subtotal'),
    ('', None, 'blank'),
    # ── NOI ──
    ('  NOI (영업이익)',    'noi', 'subtotal'),
    ('', None, 'blank'),
    # ── Debt Service ──
    ('Debt Service',       None, 'section'),
    ('  이자상환',         'interest', 'money'),
    ('  원금상환',         'principal', 'money'),
    ('  DS (이자+원금)',    'ds', 'subtotal'),
    ('  DSCR',             'dscr', 'dscr'),
    ('', None, 'blank'),
    # ── 배분 ──
    ('잔여현금 배분',      None, 'section'),
    ('  잔여현금흐름',     'cf_residual', 'subtotal'),
    ('  토큰 귀속 (1/2)',  'token_cf', 'money'),
    ('  시행사 귀속 (1/2)', 'sponsor_cf', 'money'),
    ('', None, 'blank'),
    # ── 누적수익률 ──
    ('누적수익률 (r=4.5%)', None, 'section'),
    ('  프로젝트',         'cum_return_project', 'pct'),
    ('  토큰',             'cum_return_token', 'pct'),
]


def fmt_val(val, fmt_type):
    """값을 포맷 문자열로 변환."""
    if val is None:
        return '-'
    if fmt_type == 'mwh':
        return f'{val:,.1f}'
    if fmt_type == 'price':
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
# 셀 데이터 구성 (파트별 분할)
# ============================================================
PARTS = [
    (list(range(1, 11)),  '프로젝트의 평균 현금흐름표 (1년차~10년차)',   'cashflow_table_1.png'),
    (list(range(11, 21)), '프로젝트의 평균 현금흐름표 (11년차~20년차)',  'cashflow_table_2.png'),
]

# ── 색상 정의 ──
C_HEADER   = '#2C3E50'
C_SECTION  = '#4472C4'
C_SUBTOTAL = '#D6E4F0'
C_BLANK    = '#F5F5F5'
C_DATA_ODD = '#FFFFFF'
C_DATA_EVEN = '#F2F6FA'


def build_and_render(year_range, title, filename):
    """주어진 연도 범위의 현금흐름표를 렌더링."""
    n_yr = len(year_range)
    col_labels = ['항목'] + [f'{y}년차' for y in year_range]
    n_cols = len(col_labels)

    cell_text = []
    row_formats = []

    for label, key, fmt in ROW_DEFS:
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
            val = key(d) if callable(key) else d.get(key)
            row.append(fmt_val(val, fmt))
        cell_text.append(row)
        row_formats.append(fmt)

    n_rows = len(cell_text)

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis('off')

    # 제목 (좌측)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20, loc='left')

    # 단위 표시 (테이블 오른쪽 위, 작게)
    ax.text(0.99, 1.01, '(단위: 백만원)', transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom', color='#666666')

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

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


for yr_range, title, fname in PARTS:
    build_and_render(yr_range, title, fname)

# ============================================================
# 요약 지표 출력
# ============================================================
ds_arr = np.array([data[y]['ds'] for y in years_list])
dscr_arr = np.array([data[y]['dscr'] if data[y]['dscr'] else np.nan for y in years_list])

print(f'\n=== 요약 지표 ===')
print(f'Project IRR:          {project_irr*100:.2f}%')
print(f'Token IRR:            {token_irr*100:.2f}%')
print(f'평균 DSCR (대출기간): {np.nanmean(dscr_arr[:DEBT_REPAYMENT_YEARS]):.2f}')
print(f'최소 DSCR:            {np.nanmin(dscr_arr[:DEBT_REPAYMENT_YEARS]):.2f}')
print(f'누적 토큰 배당:       {np.sum(token_arr)/1e8:.1f}억원')
