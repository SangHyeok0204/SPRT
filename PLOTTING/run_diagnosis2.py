"""
run_diagnosis2.py — 초반 우상향 원인 진단
매월 NAV 변화율 vs 토큰 변화율 비교, 롤오버 성공/실패 추적
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


def run_detailed(seed=0):
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
    model_copy = copy(model)
    model_copy.reset(seed=seed)

    records = []
    prev_nav = 0
    prev_tokens = float(INITIAL_PF_COUNT * TOKENS_PER_PF)

    for t_step in range(T):
        decision = model_copy.build_decision({'no_op': 0})
        model_copy.step(decision)

        nav = model_copy.state.NAV_t
        tokens = model_copy.state.token_count_t
        price = nav / tokens if tokens > 0 else 0

        # 변화량
        d_nav = nav - prev_nav
        d_tokens = tokens - prev_tokens
        is_rollover = (t_step + 1) % T_ROLLOVER == 0 and (t_step + 1) > 0
        rollover_happened = d_tokens > 0.5  # 토큰이 증가했으면 롤오버 성공

        active = sum(1 for pf in model_copy.pf_list if pf.is_active())
        active_pre = sum(1 for pf in model_copy.pf_list if pf.status == 'PRE' and pf.is_active())
        active_post = sum(1 for pf in model_copy.pf_list if pf.status == 'POST' and pf.is_active())

        records.append({
            't': t_step + 1,
            'nav': nav,
            'tokens': tokens,
            'price': price,
            'd_nav': d_nav,
            'd_tokens': d_tokens,
            'is_rollover_month': is_rollover,
            'rollover_success': rollover_happened if is_rollover else None,
            'active': active,
            'active_pre': active_pre,
            'active_post': active_post,
        })

        prev_nav = nav
        prev_tokens = tokens

    return records


def main():
    records = run_detailed(seed=0)

    # 롤오버 시점만 출력
    print("=" * 100)
    print(f"{'월':>5} {'NAV(억)':>10} {'토큰수':>10} {'가격':>8} "
          f"{'ΔNAV(억)':>10} {'Δ토큰':>10} {'롤오버':>8} "
          f"{'PF(활성)':>8} {'PRE':>4} {'POST':>4}")
    print("=" * 100)

    for r in records:
        # 롤오버 시점 또는 12의 배수 근처만 출력
        if r['is_rollover_month'] or r['t'] <= 24 or r['t'] % 60 == 0:
            rollover_str = ""
            if r['is_rollover_month']:
                rollover_str = "성공" if r['rollover_success'] else "실패"

            print(f"{r['t']:>5} {r['nav']/1e8:>10.1f} {r['tokens']:>10,.0f} {r['price']:>8,.0f} "
                  f"{r['d_nav']/1e8:>10.2f} {r['d_tokens']:>10,.0f} {rollover_str:>8} "
                  f"{r['active']:>8} {r['active_pre']:>4} {r['active_post']:>4}")

    # 롤오버 실패 여부 체크
    failed_rollovers = [r for r in records if r['is_rollover_month'] and not r['rollover_success']]
    print(f"\n롤오버 실패 횟수: {len(failed_rollovers)}")
    for r in failed_rollovers:
        print(f"  Month {r['t']}: NAV={r['nav']/1e8:.1f}억, tokens={r['tokens']:,.0f}")

    # NAV 증가율 vs 토큰 증가율 비교 (연간)
    print("\n=== 연간 NAV 증가율 vs 토큰 증가율 ===")
    print(f"{'연도':>4} {'NAV증가율':>10} {'토큰증가율':>10} {'가격변화율':>10} {'가격':>8}")
    for year in range(1, 40):
        idx_start = (year - 1) * 12
        idx_end = year * 12 - 1
        if idx_end >= len(records):
            break
        nav_start = records[idx_start]['nav']
        nav_end = records[idx_end]['nav']
        tok_start = records[idx_start]['tokens']
        tok_end = records[idx_end]['tokens']
        price_end = records[idx_end]['price']

        nav_growth = (nav_end / nav_start - 1) * 100 if nav_start > 0 else 0
        tok_growth = (tok_end / tok_start - 1) * 100 if tok_start > 0 else 0
        price_growth = nav_growth - tok_growth  # 근사

        print(f"{year+1:>4}년 {nav_growth:>9.1f}% {tok_growth:>9.1f}% {price_growth:>9.1f}% {price_end:>8,.0f}")


if __name__ == '__main__':
    main()
