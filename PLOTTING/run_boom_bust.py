"""
run_boom_bust.py — 산업 호황/불황 시나리오별 토큰 가격 경로

시나리오:
  1. 기본       : 매년 1개 PF 편입 (baseline)
  2. 단기 침체  : 9년차(Month 108)만 편입 0개, 이후 복귀
  3. 장기 침체  : 9~15년차(Month 108~180) 편입 0개, 이후 복귀
  4. 단기 호황  : 9년차에 3개 편입, 이후 복귀
  5. 장기 호황  : 9~15년차 편입 수 누적 증가 (2,3,4,5,6,7,8개), 이후 복귀
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from copy import copy, deepcopy

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

import SolarPFModel as M
from SolarPFModel import SolarPFModel, T_SIMULATION, T_ROLLOVER, TOKENS_PER_PF
from SolarPFPolicy import SolarPFPolicy


# ── 시나리오 정의: month → 편입 PF 수 ──

def make_schedule(name):
    """시나리오 이름 → {rollover_month: pf_count} 딕셔너리.

    딕셔너리에 없는 롤오버 시점은 기본값 1개.
    pf_count = 0 이면 해당 시점 롤오버 스킵.
    """
    schedule = {}

    if name == 'baseline':
        pass  # 전부 기본 1개

    elif name == 'short_bust':
        # 9년차(Month 108)만 편입 0개
        schedule[108] = 0

    elif name == 'long_bust':
        # 9~15년차 (Month 108, 120, ..., 180) 편입 0개 → 7년간
        for y in range(9, 16):
            schedule[y * 12] = 0

    elif name == 'short_boom':
        # 9년차에 3개 편입
        schedule[108] = 3

    elif name == 'long_boom':
        # 9~15년차: 편입 수 누적 증가 (2, 3, 4, 5, 6, 7, 8)
        for i, y in enumerate(range(9, 16)):
            schedule[y * 12] = 2 + i  # Y9=2, Y10=3, ..., Y15=8

    return schedule


def run_single_episode(schedule, seed=0):
    """커스텀 롤오버 스케줄로 단일 에피소드 실행."""
    T = T_SIMULATION

    state_names = ['time_t', 'status_t', 'P_complete_t', 'PV_t', 'NAV_t',
                   'token_count_t', 'monthly_distribution_t']
    x_names = ['no_op']
    init_state = {
        'time_t': 0, 'status_t': 'PRE', 'P_complete_t': M.P_COMPLETE_INIT,
        'PV_t': 0.0, 'NAV_t': 0.0,
        'token_count_t': float(M.INITIAL_PF_COUNT * TOKENS_PER_PF),
        'monthly_distribution_t': 0.0,
    }

    model = SolarPFModel(state_names, x_names, init_state, T, seed=seed)
    model.reset(seed=seed)

    # 원본 상수 저장
    orig_rollover_count = M.ROLLOVER_PF_COUNT

    nav_series = []
    token_series = []

    for step in range(T):
        next_month = step + 1  # model.t는 step() 내에서 +1 됨

        # 롤오버 시점이면 PF 수 오버라이드
        if next_month > 0 and next_month % T_ROLLOVER == 0:
            pf_count = schedule.get(next_month, 1)  # 기본 1개
            if pf_count == 0:
                model.no_rollover = True
                M.ROLLOVER_PF_COUNT = orig_rollover_count
            else:
                model.no_rollover = False
                M.ROLLOVER_PF_COUNT = pf_count
        else:
            model.no_rollover = False
            M.ROLLOVER_PF_COUNT = orig_rollover_count

        decision = model.build_decision({'no_op': 0})
        model.step(decision)

        nav_series.append(model.state.NAV_t)
        token_series.append(model.state.token_count_t)

    # 복원
    M.ROLLOVER_PF_COUNT = orig_rollover_count
    model.no_rollover = False

    nav_arr = np.array(nav_series)
    tok_arr = np.array(token_series)
    price_arr = nav_arr / tok_arr

    return price_arr


def run_mc(schedule, trial_size=50):
    """MC 시뮬레이션."""
    all_prices = []
    for trial in range(trial_size):
        price = run_single_episode(schedule, seed=trial)
        all_prices.append(price)
    return np.array(all_prices)


def main():
    trial_size = 1000
    T = T_SIMULATION

    scenarios = [
        ('long_bust',  '장기 침체 (9~15년차 편입 중단)',  '#b0b0b0', '--', 1.8),
        ('short_bust', '단기 침체 (9년차 편입 중단)',     '#808080', '--', 1.8),
        ('baseline',   '기본 (매년 1개)',               '#333333', '-',  2.5),
        ('short_boom', '단기 호황 (9년차 3개 편입)',      '#5b8c8c', '-',  1.8),
        ('long_boom',  '장기 호황 (9~15년차 누적 증가)',  '#2a6e6e', '-',  1.8),
    ]

    fig, ax = plt.subplots(figsize=(15, 7))
    months = np.arange(1, T + 1)

    for name, label, color, ls, lw in scenarios:
        schedule = make_schedule(name)
        print(f"  Running MC{trial_size}: {label} ...")
        all_prices = run_mc(schedule, trial_size)
        mean_price = np.mean(all_prices, axis=0)

        ax.plot(months, mean_price, color=color, linewidth=lw,
                linestyle=ls, label=label)

    # 침체/호황 구간 배경
    ax.axvspan(108, 180, color='#e0e0e0', alpha=0.3, label='침체/호황 구간 (9~15년차)')

    ax.set_xlabel('시간 (월)')
    ax.set_ylabel('토큰당 가격 (원)')
    ax.set_title('산업 호황·불황 시나리오별 평균 토큰 가격 경로')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_minor_locator(MultipleLocator(12))
    ax.set_xlim(0, T)

    plt.tight_layout()
    plt.savefig('boom_bust_scenarios.png', dpi=150, bbox_inches='tight')
    print("  -> boom_bust_scenarios.png saved")
    plt.close()


if __name__ == '__main__':
    main()
