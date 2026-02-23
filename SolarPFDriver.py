"""
SolarPFDriver.py — 몬테카를로 시뮬레이션 및 시각화

5단계 흐름: 설정 → 모델 → 정책 → MC 시뮬레이션 → 시각화
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 비대화형 백엔드 (CLI 환경용)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows 맑은고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from SolarPFModel import (
    SolarPFModel,
    T_CONSTRUCTION, T_SINGLE_PF, T_SIMULATION, T_ROLLOVER,
    P_COMPLETE_INIT, INITIAL_PF_COUNT, TOKEN_AMOUNT, TOKENS_PER_PF,
)
from SolarPFPolicy import SolarPFPolicy


def main():
    # ================================================================== #
    # ① 문제 설정                                                        #
    # ================================================================== #
    state_names = ['time_t', 'status_t', 'P_complete_t', 'PV_t', 'NAV_t', 'token_count_t', 'monthly_distribution_t']
    x_names = ['no_op']
    init_state = {
        'time_t': 0,
        'status_t': 'PRE',
        'P_complete_t': P_COMPLETE_INIT,
        'PV_t': 0.0,
        'NAV_t': 0.0,
        'token_count_t': float(INITIAL_PF_COUNT * TOKENS_PER_PF),  # 5 PF × 120,000 = 600,000 토큰
        'monthly_distribution_t': 0.0,  # 월별 분배금 (건설 중에는 0)
    }
    T = T_SIMULATION      # 480개월 (40년, 롤오버 포함)
    trial_size = 50

    # ================================================================== #
    # ② 모델 생성                                                        #
    # ================================================================== #
    model = SolarPFModel(
        state_names, x_names, init_state, T,
        risk_adjust='YES', seed=0,
    )

    # ================================================================== #
    # ③ 정책 생성                                                        #
    # ================================================================== #
    policy = SolarPFPolicy(model, policy_type='no_op')

    # ================================================================== #
    # ④ 몬테카를로 시뮬레이션                                             #
    # ================================================================== #
    all_nav = []
    all_pv = []
    all_p_complete = []
    all_status = []
    all_token_count = []
    all_distribution = []  # 월별 분배금 (펀드 전체)

    print(f"Running {trial_size} Monte Carlo simulations (T={T} months)...")

    for trial in range(trial_size):
        _, history = policy.run_policy(seed=trial)

        all_nav.append([h['state'].NAV_t for h in history])
        all_pv.append([h['state'].PV_t for h in history])
        all_p_complete.append([h['state'].P_complete_t for h in history])
        all_status.append([h['state'].status_t for h in history])
        all_token_count.append([h['state'].token_count_t for h in history])
        all_distribution.append([h['state'].monthly_distribution_t for h in history])

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{trial_size} done")

    all_nav = np.array(all_nav)            # (trial_size, T)
    all_pv = np.array(all_pv)
    all_p_complete = np.array(all_p_complete)
    all_token_count = np.array(all_token_count)
    all_distribution = np.array(all_distribution)

    # 토큰당 가격 계산 (NAV / token_count)
    all_price_per_token = all_nav / all_token_count

    # 토큰당 분배금 계산 (distribution / token_count)
    all_dist_per_token = all_distribution / all_token_count

    # ================================================================== #
    # ⑤ 시각화                                                           #
    # ================================================================== #
    days = np.arange(1, T + 1)
    cmap = plt.cm.hsv
    colors = [cmap(i / trial_size) for i in range(trial_size)]

    fig, axes = plt.subplots(4, 1, figsize=(15, 17), sharex=True)

    # 억 원 단위 변환 (1억 = 1e8)
    UNIT = 1e8
    UNIT_LABEL = '억 원'

    # 롤오버 시점 계산
    rollover_months = [t for t in range(T_ROLLOVER, T + 1, T_ROLLOVER)]

    # --- NAV 경로 (펀드 전체 가치) ---
    ax = axes[0]
    for i in range(trial_size):
        ax.plot(days, all_nav[i] / UNIT, color=colors[i], alpha=0.35, linewidth=0.6)
    mean_nav = np.mean(all_nav, axis=0)
    ax.plot(days, mean_nav / UNIT, color='black', linewidth=1.8, label='Mean NAV')
    ax.axvline(x=T_CONSTRUCTION, color='red', linestyle='--',
               alpha=0.7, linewidth=1.2, label=f'Completion (Month {T_CONSTRUCTION})')
    # 롤오버 시점 표시
    for i, rm in enumerate(rollover_months):
        label = 'Rollover' if i == 0 else None
        ax.axvline(x=rm, color='green', linestyle=':', alpha=0.5, linewidth=1.0, label=label)
    ax.set_ylabel(f'Total NAV ({UNIT_LABEL})')
    ax.set_title(f'Solar PF Fund Total NAV ({INITIAL_PF_COUNT} PFs) — {trial_size} MC Paths')
    ax.legend(loc='upper right')

    # --- 토큰당 가격 (NAV / token_count) ---
    ax = axes[1]
    for i in range(trial_size):
        ax.plot(days, all_price_per_token[i] / UNIT, color=colors[i], alpha=0.35, linewidth=0.6)
    mean_price = np.mean(all_price_per_token, axis=0)
    ax.plot(days, mean_price / UNIT, color='black', linewidth=1.8, label='Mean Price/Token')
    ax.axvline(x=T_CONSTRUCTION, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
    for i, rm in enumerate(rollover_months):
        label = 'Rollover' if i == 0 else None
        ax.axvline(x=rm, color='green', linestyle=':', alpha=0.5, linewidth=1.0, label=label)
    ax.set_ylabel(f'Price per Token ({UNIT_LABEL})')
    ax.set_title('Token Price = Total NAV / Token Count')
    ax.legend(loc='upper right')

    # --- PV 경로 ---
    ax = axes[2]
    for i in range(trial_size):
        ax.plot(days, all_pv[i] / UNIT, color=colors[i], alpha=0.35, linewidth=0.6)
    mean_pv = np.mean(all_pv, axis=0)
    ax.plot(days, mean_pv / UNIT, color='black', linewidth=1.8, label='Mean PV')
    ax.axvline(x=T_CONSTRUCTION, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
    for rm in rollover_months:
        ax.axvline(x=rm, color='green', linestyle=':', alpha=0.5, linewidth=1.0)
    ax.set_ylabel(f'PV ({UNIT_LABEL})')
    ax.set_title('Present Value of Future Cash Flows (Total Fund)')
    ax.legend(loc='upper right')

    # --- 완공 확률 (평균) ---
    ax = axes[3]
    for i in range(trial_size):
        ax.plot(days, all_p_complete[i], color=colors[i], alpha=0.35, linewidth=0.6)
    mean_p = np.mean(all_p_complete, axis=0)
    ax.plot(days, mean_p, color='black', linewidth=1.8, label='Mean P(complete)')
    ax.axvline(x=T_CONSTRUCTION, color='red', linestyle='--', alpha=0.7, linewidth=1.2)
    for rm in rollover_months:
        ax.axvline(x=rm, color='green', linestyle=':', alpha=0.5, linewidth=1.0)
    ax.set_ylabel('P(complete)')
    ax.set_xlabel('Time (months)')
    ax.set_title('Average Completion Probability (PRE state PFs)')
    ax.legend(loc='lower right')

    # X축: 60개월(5년) 주눈금 + 12개월(1년) 보조눈금
    from matplotlib.ticker import MultipleLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.xaxis.set_minor_locator(MultipleLocator(12))
        ax.tick_params(axis='x', which='minor', length=3)
        ax.tick_params(axis='x', which='major', length=6)
        ax.set_xlim(0, T)

    plt.tight_layout()
    plt.savefig('nav_simulation_result.png', dpi=150, bbox_inches='tight')
    print("Chart saved to nav_simulation_result.png")

    # ================================================================== #
    # ⑥ 월별 분배금 막대 그래프 (별도 Figure)                             #
    # ================================================================== #
    fig2, ax2 = plt.subplots(figsize=(15, 6))

    # 평균 토큰당 분배금 계산
    mean_dist_per_token = np.mean(all_dist_per_token, axis=0)

    # 12개월 단위로 합산하여 연간 분배금 계산
    years = T // 12
    annual_dist_per_token = []
    for year in range(years):
        start_idx = year * 12
        end_idx = start_idx + 12
        annual_sum = np.sum(mean_dist_per_token[start_idx:end_idx])
        annual_dist_per_token.append(annual_sum)

    # 막대 그래프 (연간 분배금)
    year_labels = [f'Y{y+1}' for y in range(years)]
    bars = ax2.bar(range(years), annual_dist_per_token, color='steelblue', alpha=0.8, edgecolor='black')

    # 건설기간 표시 (1년차는 건설 중이므로 분배 없음)
    ax2.axvspan(-0.5, 0.5, color='red', alpha=0.2, label='건설기간 (분배 없음)')

    # 롤오버 시점 표시
    for i, rm in enumerate(rollover_months):
        rollover_year = rm // 12 - 0.5  # 막대 중심에 맞춤
        if 0 <= rollover_year < years:
            label = '롤오버 시점' if i == 0 else None
            ax2.axvline(x=rollover_year, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label=label)

    ax2.set_xlabel('연도')
    ax2.set_ylabel('연간 분배금 (원/토큰)')
    ax2.set_title(f'토큰당 연간 분배금 (평균, {trial_size} MC 시뮬레이션)')
    ax2.set_xticks(range(0, years, 5))
    ax2.set_xticklabels([f'Y{y+1}' for y in range(0, years, 5)])
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)

    # 평균 연간 분배금 수평선
    avg_annual = np.mean(annual_dist_per_token[1:])  # 건설기간 제외
    ax2.axhline(y=avg_annual, color='orange', linestyle='-', linewidth=2,
                label=f'평균 연간 분배금: {avg_annual:,.0f}원')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('distribution_per_token.png', dpi=150, bbox_inches='tight')
    print("Chart saved to distribution_per_token.png")

    # ================================================================== #
    # ⑦ 월별 분배금 상세 그래프 (첫 10년)                                 #
    # ================================================================== #
    fig3, ax3 = plt.subplots(figsize=(15, 5))

    # 첫 120개월 (10년)만 표시
    display_months = min(120, T)
    months_range = np.arange(1, display_months + 1)
    mean_monthly_dist = mean_dist_per_token[:display_months]

    ax3.bar(months_range, mean_monthly_dist, color='teal', alpha=0.7, width=0.8)
    ax3.axvline(x=T_CONSTRUCTION, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
                label=f'완공 시점 (Month {T_CONSTRUCTION})')
    ax3.axvline(x=T_ROLLOVER, color='green', linestyle=':', alpha=0.7, linewidth=1.5,
                label=f'1차 롤오버 (Month {T_ROLLOVER})')

    ax3.set_xlabel('월 (Month)')
    ax3.set_ylabel('월별 분배금 (원/토큰)')
    ax3.set_title(f'토큰당 월별 분배금 (첫 {display_months//12}년, 평균)')
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)

    # X축 12개월 단위 눈금
    ax3.xaxis.set_major_locator(MultipleLocator(12))
    ax3.xaxis.set_minor_locator(MultipleLocator(6))
    ax3.set_xlim(0, display_months + 1)

    plt.tight_layout()
    plt.savefig('monthly_distribution_detail.png', dpi=150, bbox_inches='tight')
    print("Chart saved to monthly_distribution_detail.png")

    # ================================================================== #
    # 요약 통계                                                           #
    # ================================================================== #
    failed_count = sum(
        1 for trial_status in all_status if 'FAILED' in trial_status
    )
    final_navs = all_nav[:, -1]
    peak_navs = np.max(all_nav, axis=1)
    final_prices = all_price_per_token[:, -1]
    final_token_counts = all_token_count[:, -1]

    # 분배금 통계 계산
    total_dist_per_token = np.sum(all_dist_per_token, axis=1)  # 40년간 총 분배금
    mean_annual_dist = np.mean(np.sum(all_dist_per_token[:, 12:], axis=1) / (years - 1))  # 연평균 (건설기간 제외)

    print("\n========== Simulation Results ==========")
    print(f"Trials           : {trial_size}")
    print(f"Horizon          : {T} months ({T/12:.1f} years)")
    print(f"Failed projects  : {failed_count}/{trial_size}")
    print(f"Mean final NAV   : {np.mean(final_navs):>15,.0f} KRW")
    print(f"Std  final NAV   : {np.std(final_navs):>15,.0f} KRW")
    print(f"Mean peak NAV    : {np.mean(peak_navs):>15,.0f} KRW")
    print(f"Std  peak NAV    : {np.std(peak_navs):>15,.0f} KRW")
    print("-" * 40)
    print(f"Mean final token count : {np.mean(final_token_counts):>10,.2f}")
    print(f"Mean final price/token : {np.mean(final_prices):>15,.0f} KRW")
    print(f"Std  final price/token : {np.std(final_prices):>15,.0f} KRW")
    print("-" * 40)
    print(f"Mean total dist/token (40yr) : {np.mean(total_dist_per_token):>12,.0f} KRW")
    print(f"Mean annual dist/token       : {mean_annual_dist:>12,.0f} KRW")
    print(f"Implied yield (dist/price)   : {(mean_annual_dist / 10000) * 100:>10.2f} %")
    print("========================================")


if __name__ == '__main__':
    main()
