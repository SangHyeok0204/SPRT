"""
run_default_analysis.py — 부도 관련 민감도 분석 플롯 2종

Plot 1: 초기 완공확률(P_COMPLETE_INIT) 민감도  (0.90 / 0.95 / 0.99)
Plot 2: 월별 부도확률(P_DEFAULT) 민감도        (0.003 / 0.005 / 0.008)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from copy import copy

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

import SolarPFModel as M
from SolarPFModel import SolarPFModel, T_SIMULATION, T_CONSTRUCTION, T_ROLLOVER, TOKENS_PER_PF
from SolarPFPolicy import SolarPFPolicy


def run_mc(trial_size, T, p_complete_init=None, p_default=None):
    """MC 시뮬레이션 실행. 모듈 상수를 임시 교체 후 복원."""
    # 원본 저장
    orig_p_complete = M.P_COMPLETE_INIT
    orig_p_default = M.P_DEFAULT

    # 파라미터 교체
    if p_complete_init is not None:
        M.P_COMPLETE_INIT = p_complete_init
    if p_default is not None:
        M.P_DEFAULT = p_default

    state_names = ['time_t', 'status_t', 'P_complete_t', 'PV_t', 'NAV_t',
                   'token_count_t', 'monthly_distribution_t']
    x_names = ['no_op']
    init_state = {
        'time_t': 0,
        'status_t': 'PRE',
        'P_complete_t': M.P_COMPLETE_INIT,
        'PV_t': 0.0,
        'NAV_t': 0.0,
        'token_count_t': float(M.INITIAL_PF_COUNT * TOKENS_PER_PF),
        'monthly_distribution_t': 0.0,
    }

    model = SolarPFModel(state_names, x_names, init_state, T, seed=0)
    policy = SolarPFPolicy(model, policy_type='no_op')

    all_nav = []
    all_token_count = []
    all_status = []

    for trial in range(trial_size):
        _, history, _ = policy.run_policy(seed=trial)
        all_nav.append([h['state'].NAV_t for h in history])
        all_token_count.append([h['state'].token_count_t for h in history])
        all_status.append([h['state'].status_t for h in history])

    # 원본 복원
    M.P_COMPLETE_INIT = orig_p_complete
    M.P_DEFAULT = orig_p_default

    all_nav = np.array(all_nav)
    all_token_count = np.array(all_token_count)
    all_price = all_nav / all_token_count

    # 부도 발생 trial 수 계산
    failed_count = sum(1 for s in all_status if 'FAILED' in s)

    return all_price, failed_count


def plot_p_complete_sensitivity():
    """Plot 1: 초기 완공확률 민감도 (0.90 / 0.95 / 0.99)"""
    T = T_SIMULATION
    trial_size = 100
    scenarios = [
        (0.90, 'P_complete = 0.90 (건전성 악화)', '#e74c3c'),
        (0.95, 'P_complete = 0.95 (기본)', '#2c3e50'),
        (0.99, 'P_complete = 0.99 (건전성 개선)', '#2ecc71'),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    days = np.arange(1, T + 1)

    for p_init, label, color in scenarios:
        print(f"  Running MC{trial_size} with P_COMPLETE_INIT={p_init} ...")
        all_price, failed = run_mc(trial_size, T, p_complete_init=p_init)
        mean_price = np.mean(all_price, axis=0)

        ax.plot(days, mean_price, color=color, linewidth=2.0,
                label=f'{label}  (부도 {failed}/{trial_size})')

    ax.set_xlabel('시간 (월)')
    ax.set_ylabel('토큰당 가격 (원)')
    ax.set_title(f'초기 완공확률별 평균 토큰 가격 경로 (MC {trial_size}회)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(12))
    ax.xaxis.set_minor_locator(MultipleLocator(6))
    ax.set_xlim(0, 48)

    plt.tight_layout()
    plt.savefig('sensitivity_p_complete.png', dpi=150, bbox_inches='tight')
    print("  -> sensitivity_p_complete.png saved")
    plt.close()


def plot_p_default_sensitivity():
    """Plot 2: 월별 부도확률 민감도 (0.003 / 0.005 / 0.008)"""
    T = T_SIMULATION
    trial_size = 100
    scenarios = [
        (0.003, 'P_default = 0.3%/월 (기본)', '#2c3e50'),
        (0.005, 'P_default = 0.5%/월', '#e67e22'),
        (0.008, 'P_default = 0.8%/월', '#e74c3c'),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    days = np.arange(1, T + 1)

    for p_def, label, color in scenarios:
        # 건설기간 부도확률 계산
        prob_fail = 1 - (1 - p_def) ** 12
        print(f"  Running MC{trial_size} with P_DEFAULT={p_def} "
              f"(건설기간 부도확률 {prob_fail:.1%}) ...")
        all_price, failed = run_mc(trial_size, T, p_default=p_def)
        mean_price = np.mean(all_price, axis=0)

        ax.plot(days, mean_price, color=color, linewidth=2.0,
                label=f'{label}  (건설기간 부도 {prob_fail:.1%}, 발생 {failed}/{trial_size})')

    ax.set_xlabel('시간 (월)')
    ax.set_ylabel('토큰당 가격 (원)')
    ax.set_title(f'부도확률별 평균 토큰 가격 경로 (MC {trial_size}회)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(12))
    ax.set_xlim(0, T)

    plt.tight_layout()
    plt.savefig('sensitivity_p_default.png', dpi=150, bbox_inches='tight')
    print("  -> sensitivity_p_default.png saved")
    plt.close()


if __name__ == '__main__':
    print("=" * 50)
    print("Plot 1: 초기 완공확률 민감도")
    print("=" * 50)
    plot_p_complete_sensitivity()

    print()
    print("=" * 50)
    print("Plot 2: 건설 부도확률 민감도")
    print("=" * 50)
    plot_p_default_sensitivity()

    print("\nDone.")
