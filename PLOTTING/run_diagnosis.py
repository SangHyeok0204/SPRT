"""
run_diagnosis.py — 토큰 가격 우하향 원인 진단
NAV, 토큰 수, 활성 PF 수, PF별 PV 기여분 추적
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from copy import copy

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from SolarPFModel import (
    SolarPFModel, T_SIMULATION, T_CONSTRUCTION, T_ROLLOVER,
    T_SINGLE_PF, TOKENS_PER_PF, P_COMPLETE_INIT, INITIAL_PF_COUNT,
)
from SolarPFPolicy import SolarPFPolicy


def run_single_diagnostic(seed=0):
    """단일 시드로 상세 진단 데이터 추출."""
    state_names = ['time_t', 'status_t', 'P_complete_t', 'PV_t', 'NAV_t',
                   'token_count_t', 'monthly_distribution_t']
    x_names = ['no_op']
    init_state = {
        'time_t': 0, 'status_t': 'PRE', 'P_complete_t': P_COMPLETE_INIT,
        'PV_t': 0.0, 'NAV_t': 0.0,
        'token_count_t': float(INITIAL_PF_COUNT * TOKENS_PER_PF),
        'monthly_distribution_t': 0.0,
    }
    T = T_SIMULATION

    model = SolarPFModel(state_names, x_names, init_state, T, seed=seed)
    policy = SolarPFPolicy(model, policy_type='no_op')

    # 직접 에피소드 실행 (내부 상태 접근 위해)
    model_copy = copy(model)
    model_copy.reset(seed=seed)

    nav_list = []
    token_list = []
    price_list = []
    active_pf_list = []
    rollover_events = []

    for t_step in range(T):
        decision = model_copy.build_decision({'no_op': 0})
        model_copy.step(decision)

        nav = model_copy.state.NAV_t
        tokens = model_copy.state.token_count_t
        price = nav / tokens if tokens > 0 else 0

        active = sum(1 for pf in model_copy.pf_list if pf.is_active())

        nav_list.append(nav)
        token_list.append(tokens)
        price_list.append(price)
        active_pf_list.append(active)

    return {
        'nav': np.array(nav_list),
        'tokens': np.array(token_list),
        'price': np.array(price_list),
        'active_pf': np.array(active_pf_list),
        'rollover_schedule': model_copy.rollover_schedule,
    }


def main():
    data = run_single_diagnostic(seed=0)
    T = T_SIMULATION
    months = np.arange(1, T + 1)

    print("=== 롤오버 이력 ===")
    print(f"{'시점':>6} {'신규토큰':>12} {'발행가':>10} {'증자후가격':>12}")
    for r in data['rollover_schedule']:
        print(f"{r['t']:>6} {r['new_tokens']:>12,.0f} {r['issue_price']:>10,.0f} {r['token_price']:>12,.0f}")

    print(f"\n=== 주요 시점 요약 ===")
    for m in [12, 60, 120, 180, 240, 300, 360, 480]:
        idx = min(m - 1, T - 1)
        print(f"Month {m:>3}: NAV={data['nav'][idx]:>15,.0f}  "
              f"Tokens={data['tokens'][idx]:>10,.0f}  "
              f"Price={data['price'][idx]:>8,.0f}  "
              f"Active PFs={data['active_pf'][idx]:>3}")

    # 4-panel 진단 차트
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    UNIT = 1e8

    # 1. Total NAV
    ax = axes[0]
    ax.plot(months, data['nav'] / UNIT, color='#2c3e50', linewidth=1.5)
    ax.set_ylabel('Total NAV (억 원)')
    ax.set_title('펀드 전체 NAV')
    ax.grid(alpha=0.3)

    # 2. Total Token Count
    ax = axes[1]
    ax.plot(months, data['tokens'], color='#e67e22', linewidth=1.5)
    ax.set_ylabel('총 토큰 수')
    ax.set_title('누적 발행 토큰 수')
    ax.grid(alpha=0.3)

    # 3. Token Price = NAV / Tokens
    ax = axes[2]
    ax.plot(months, data['price'], color='#27ae60', linewidth=1.5)
    ax.set_ylabel('토큰당 가격 (원)')
    ax.set_title('토큰 가격 = NAV / 토큰 수')
    ax.grid(alpha=0.3)

    # 4. Active PF Count
    ax = axes[3]
    ax.plot(months, data['active_pf'], color='#8e44ad', linewidth=1.5)
    ax.set_ylabel('활성 PF 수')
    ax.set_xlabel('시간 (월)')
    ax.set_title('활성 PF 개수')
    ax.grid(alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(60))
        ax.xaxis.set_minor_locator(MultipleLocator(12))
        ax.set_xlim(0, T)

    plt.tight_layout()
    plt.savefig('diagnosis_nav_tokens.png', dpi=150, bbox_inches='tight')
    print("\n-> diagnosis_nav_tokens.png saved")
    plt.close()


if __name__ == '__main__':
    main()
