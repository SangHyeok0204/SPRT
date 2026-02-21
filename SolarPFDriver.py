"""
SolarPFDriver.py — 몬테카를로 시뮬레이션 및 시각화

5단계 흐름: 설정 → 모델 → 정책 → MC 시뮬레이션 → 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows 맑은고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from SolarPFModel import (
    SolarPFModel,
    T_CONSTRUCTION, T_SINGLE_PF,
    P_COMPLETE_INIT,
)
from SolarPFPolicy import SolarPFPolicy


def main():
    # ================================================================== #
    # ① 문제 설정                                                        #
    # ================================================================== #
    state_names = ['time_t', 'status_t', 'P_complete_t', 'PV_t', 'NAV_t']
    x_names = ['no_op']
    init_state = {
        'time_t': 0,
        'status_t': 'PRE',
        'P_complete_t': P_COMPLETE_INIT,
        'PV_t': 0.0,
        'NAV_t': 0.0,
    }
    T = T_SINGLE_PF       # 8030일 (단일 PF 생애주기)
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

    print(f"Running {trial_size} Monte Carlo simulations (T={T} months)...")

    for trial in range(trial_size):
        _, history = policy.run_policy(seed=trial)

        all_nav.append([h['state'].NAV_t for h in history])
        all_pv.append([h['state'].PV_t for h in history])
        all_p_complete.append([h['state'].P_complete_t for h in history])
        all_status.append([h['state'].status_t for h in history])

        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{trial_size} done")

    all_nav = np.array(all_nav)            # (trial_size, T)
    all_pv = np.array(all_pv)
    all_p_complete = np.array(all_p_complete)

    # ================================================================== #
    # ⑤ 시각화                                                           #
    # ================================================================== #
    days = np.arange(1, T + 1)
    cmap = plt.cm.hsv
    colors = [cmap(i / trial_size) for i in range(trial_size)]

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

    # 억 원 단위 변환 (1억 = 1e8)
    UNIT = 1e8
    UNIT_LABEL = '억 원'

    # --- NAV 경로 ---
    ax = axes[0]
    for i in range(trial_size):
        ax.plot(days, all_nav[i] / UNIT, color=colors[i], alpha=0.35, linewidth=0.6)
    mean_nav = np.mean(all_nav, axis=0)
    ax.plot(days, mean_nav / UNIT, color='black', linewidth=1.8, label='Mean NAV')
    ax.axvline(x=T_CONSTRUCTION, color='red', linestyle='--',
               alpha=0.7, linewidth=1.2, label=f'Completion (Month {T_CONSTRUCTION})')
    ax.set_ylabel(f'NAV ({UNIT_LABEL})')
    ax.set_title(f'Solar PF NAV Simulation — {trial_size} Monte Carlo Paths')
    ax.legend(loc='upper right')

    # --- PV 경로 ---
    ax = axes[1]
    for i in range(trial_size):
        ax.plot(days, all_pv[i] / UNIT, color=colors[i], alpha=0.35, linewidth=0.6)
    mean_pv = np.mean(all_pv, axis=0)
    ax.plot(days, mean_pv / UNIT, color='black', linewidth=1.8, label='Mean PV')
    ax.axvline(x=T_CONSTRUCTION, color='red', linestyle='--',
               alpha=0.7, linewidth=1.2)
    ax.set_ylabel(f'PV ({UNIT_LABEL})')
    ax.set_title('Present Value of Future Cash Flows')
    ax.legend(loc='upper right')

    # --- 완공 확률 ---
    ax = axes[2]
    for i in range(trial_size):
        ax.plot(days, all_p_complete[i], color=colors[i], alpha=0.35, linewidth=0.6)
    mean_p = np.mean(all_p_complete, axis=0)
    ax.plot(days, mean_p, color='black', linewidth=1.8, label='Mean P(complete)')
    ax.axvline(x=T_CONSTRUCTION, color='red', linestyle='--',
               alpha=0.7, linewidth=1.2)
    ax.set_ylabel('P(complete)')
    ax.set_xlabel('Time (months)')
    ax.set_title('Completion Probability')
    ax.legend(loc='lower right')

    # X축: 500일 주눈금 라벨 + 100일 보조눈금
    from matplotlib.ticker import MultipleLocator
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(24))
        ax.xaxis.set_minor_locator(MultipleLocator(6))
        ax.tick_params(axis='x', which='minor', length=3)
        ax.tick_params(axis='x', which='major', length=6)
        ax.set_xlim(0, T)

    plt.tight_layout()
    plt.savefig('nav_simulation_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ================================================================== #
    # 요약 통계                                                           #
    # ================================================================== #
    failed_count = sum(
        1 for trial_status in all_status if 'FAILED' in trial_status
    )
    final_navs = all_nav[:, -1]
    peak_navs = np.max(all_nav, axis=1)

    print("\n========== Simulation Results ==========")
    print(f"Trials           : {trial_size}")
    print(f"Horizon          : {T} months ({T/12:.1f} years)")
    print(f"Failed projects  : {failed_count}/{trial_size}")
    print(f"Mean final NAV   : {np.mean(final_navs):>15,.0f} KRW")
    print(f"Std  final NAV   : {np.std(final_navs):>15,.0f} KRW")
    print(f"Mean peak NAV    : {np.mean(peak_navs):>15,.0f} KRW")
    print(f"Std  peak NAV    : {np.std(peak_navs):>15,.0f} KRW")
    print("========================================")


if __name__ == '__main__':
    main()
