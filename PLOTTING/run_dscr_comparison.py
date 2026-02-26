"""
run_dscr_comparison.py — SPRT 토큰화 비율별 DSCR 개선 효과 비교

자본구조 (총 사업비 20억, 자기자본 4억 고정):
  - SPRT  0%: 자기자본 4억 + 대출 16억 + 토큰 0   (전통적 PF)
  - SPRT 30%: 자기자본 4억 + 대출 10억 + 토큰 6억
  - SPRT 60%: 자기자본 4억 + 대출  4억 + 토큰 12억

DSCR = NOI / Debt Service
  NOI = 월 매출액 - 고정비용 (OPEX + 사무위탁 + 대수선)
  Debt Service = 원금상환 + 이자비용
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 공통 파라미터 ──
CAPITAL_TOTAL = 2_000_000_000       # 총 사업비 20억
EQUITY = 400_000_000                # 자기자본 4억 (고정)
DEBT_PLUS_TOKEN = CAPITAL_TOTAL - EQUITY  # 외부조달 16억

INTEREST_RATE = 0.06                # 선순위 대출 연이자율
REPAYMENT_YEARS = 15                # 대출 상환기간
REPAYMENT_MONTHS = REPAYMENT_YEARS * 12

OPEX_ANNUAL = 10_000_000
ADMIN_FEE_ANNUAL = 10_000_000
MAJOR_REPAIR_ANNUAL = 5_000_000
FIXED_COST_MONTHLY = (OPEX_ANNUAL + ADMIN_FEE_ANNUAL + MAJOR_REPAIR_ANNUAL) / 12

# 매출 파라미터
FIXED_PRICE = 113.0 + 77.0         # SMP + REC 고정가격 (원/kWh)
BASE_GENERATION = 105.9             # 월 평균 발전량 (MWh)
DEGRADATION_ANNUAL = 0.005          # 연 0.5% 열화

T_OPERATION = 240                   # 운영기간 20년


def compute_annual_dscr(token_ratio):
    """토큰화 비율에 따른 연간 DSCR 계산 (운영기간 20년).

    Parameters
    ----------
    token_ratio : float
        토큰화 비율 (0.0 = 전통PF, 0.3 = 30%, 0.6 = 60%)

    Returns
    -------
    years : array, dscr_values : array
    """
    token_amount = DEBT_PLUS_TOKEN * token_ratio
    debt_amount = DEBT_PLUS_TOKEN - token_amount

    # 원리금균등상환 월 납입액
    r_m = INTEREST_RATE / 12
    n_months = REPAYMENT_MONTHS
    if debt_amount > 0:
        monthly_pmt = debt_amount * r_m * (1 + r_m) ** n_months / ((1 + r_m) ** n_months - 1)
    else:
        monthly_pmt = 0.0

    annual_dscr = []

    for year in range(1, 21):  # 운영 1~20년차
        monthly_noi_list = []
        monthly_ds_list = []

        for m in range(12):
            op_month = (year - 1) * 12 + m  # 운영 경과 월수

            # 매출 (열화 반영)
            deg_factor = (1 - DEGRADATION_ANNUAL) ** (op_month / 12)
            revenue = FIXED_PRICE * BASE_GENERATION * 1000 * deg_factor

            # NOI = 매출 - 고정비용
            noi = revenue - FIXED_COST_MONTHLY

            # Debt Service (원리금균등상환, 상환기간 내만)
            if op_month < REPAYMENT_MONTHS and debt_amount > 0:
                ds = monthly_pmt
            else:
                ds = 0.0

            monthly_noi_list.append(noi)
            monthly_ds_list.append(ds)

        annual_noi = sum(monthly_noi_list)
        annual_ds = sum(monthly_ds_list)

        if annual_ds > 0:
            dscr = annual_noi / annual_ds
        else:
            dscr = None  # 상환 완료 → DSCR 무한대

        annual_dscr.append(dscr)

    return annual_dscr


def main():
    scenarios = [
        (0.00, 'SPRT 0% (전통적 PF)', '#e74c3c', '--'),
        (0.30, 'SPRT 30%', '#e67e22', '-'),
        (0.60, 'SPRT 60%', '#2c3e50', '-'),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    for token_ratio, label, color, ls in scenarios:
        dscr_list = compute_annual_dscr(token_ratio)

        debt = DEBT_PLUS_TOKEN * (1 - token_ratio)
        token = DEBT_PLUS_TOKEN * token_ratio

        # 상환기간(15년)까지만 DSCR 표시
        years_plot = []
        dscr_plot = []
        for i, d in enumerate(dscr_list):
            if d is not None:
                years_plot.append(i + 1)
                dscr_plot.append(d)

        lbl = f'{label}  (대출 {debt/1e8:.0f}억, 토큰 {token/1e8:.0f}억)'
        ax.plot(years_plot, dscr_plot, color=color, linewidth=2.5,
                linestyle=ls, marker='o', markersize=4, label=lbl)

        # 첫해, 15년차 DSCR 값 표시
        if dscr_plot:
            ax.annotate(f'{dscr_plot[0]:.2f}', xy=(years_plot[0], dscr_plot[0]),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold')
            ax.annotate(f'{dscr_plot[-1]:.2f}', xy=(years_plot[-1], dscr_plot[-1]),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold')

    # DSCR 기준선
    ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
               label='DSCR = 1.0 (손익분기)')
    ax.axhline(y=1.3, color='orange', linestyle=':', linewidth=1.0, alpha=0.5,
               label='DSCR = 1.3 (금융기관 통상 요구)')

    ax.set_xlabel('운영 연차')
    ax.set_ylabel('DSCR')
    ax.set_title('토큰화 비율별 DSCR 비교 (단일 PF, 대출 상환기간 15년)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlim(0.5, 15.5)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig('dscr_comparison.png', dpi=150, bbox_inches='tight')
    print("-> dscr_comparison.png saved")
    plt.close()

    # 요약 통계
    print("\n=== DSCR 요약 (운영 1년차 / 15년차) ===")
    for token_ratio, label, _, _ in scenarios:
        dscr_list = compute_annual_dscr(token_ratio)
        debt = DEBT_PLUS_TOKEN * (1 - token_ratio)
        valid = [d for d in dscr_list if d is not None]
        print(f"  {label:20s}  대출 {debt/1e8:4.0f}억 | "
              f"1년차 DSCR={valid[0]:.2f}  15년차 DSCR={valid[-1]:.2f}  "
              f"평균 DSCR={np.mean(valid):.2f}")


if __name__ == '__main__':
    main()
