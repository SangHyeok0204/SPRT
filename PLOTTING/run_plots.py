"""
run_plots.py — 논문용 시뮬레이션 플롯 일괄 생성

실행: python run_plots.py
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
    SolarPFModel,
    T_CONSTRUCTION, T_OPERATION, T_SIMULATION, T_SINGLE_PF,
    P_COMPLETE_INIT, INITIAL_PF_COUNT, TOKENS_PER_PF,
)
from SolarPFPolicy import SolarPFPolicy

# ── 공통 설정 ──
STATE_NAMES = [
    'time_t', 'status_t', 'P_complete_t',
    'PV_t', 'NAV_t', 'token_count_t', 'monthly_distribution_t',
]
X_NAMES = ['no_op']

def make_init_state(p_complete=P_COMPLETE_INIT):
    return {
        'time_t': 0,
        'status_t': 'PRE',
        'P_complete_t': p_complete,
        'PV_t': 0.0,
        'NAV_t': 0.0,
        'token_count_t': float(INITIAL_PF_COUNT * TOKENS_PER_PF),
        'monthly_distribution_t': 0.0,
    }


def run_mc(T, trial_size, no_rollover=False, no_merge=False, p_complete=P_COMPLETE_INIT):
    """MC 시뮬레이션 실행, 결과 dict 반환."""
    p_override = p_complete if p_complete != P_COMPLETE_INIT else None
    model = SolarPFModel(
        STATE_NAMES, X_NAMES, make_init_state(p_complete), T,
        seed=0, no_rollover=no_rollover, no_merge=no_merge,
        p_complete_override=p_override,
    )
    policy = SolarPFPolicy(model, policy_type='no_op')

    all_nav, all_tokens, all_merge_hist = [], [], []
    all_pf_events = []  # 부도 이벤트 추적용

    for trial in range(trial_size):
        model_copy = copy(model)
        model_copy.reset(seed=trial)
        policy_copy = SolarPFPolicy(model_copy, 'no_op')
        for _ in range(model_copy.T):
            decision = policy_copy.get_decision()
            model_copy.step(decision)

        all_nav.append([h['state'].NAV_t for h in model_copy.history])
        all_tokens.append([h['state'].token_count_t for h in model_copy.history])
        all_merge_hist.append(model_copy.merge_history.copy())

        # 부도 PF 탐지
        failed_pfs = [pf for pf in model_copy.pf_list if pf.status == 'FAILED']
        all_pf_events.append(failed_pfs)

    all_nav = np.array(all_nav)
    all_tokens = np.array(all_tokens)
    all_price = all_nav / all_tokens

    return {
        'nav': all_nav,
        'tokens': all_tokens,
        'price': all_price,
        'merge_hist': all_merge_hist,
        'pf_events': all_pf_events,
    }


# ================================================================== #
# Plot 1 & 2: 병합 유/무 평균 가격 경로                               #
# ================================================================== #
def plot_merge_comparison():
    print("=== Plot 1&2: 병합 유/무 평균 가격 비교 ===")
    T = T_SIMULATION
    N = 100

    print("  병합 없이 MC100...")
    res_no = run_mc(T, N, no_merge=True)
    print("  병합 있이 MC100...")
    res_yes = run_mc(T, N, no_merge=False)

    years = np.arange(1, T + 1) / 12
    mean_no = np.mean(res_no['price'], axis=0)
    mean_yes = np.mean(res_yes['price'], axis=0)

    # Plot 1: 병합 없을 때
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(years, mean_no, color='black', linewidth=2)
    ax.set_xlabel('경과 연수')
    ax.set_ylabel('토큰 가격 (원)')
    ax.set_title('토큰 병합 없을 때 평균 가격 경로')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0, T / 12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_price_no_merge.png', dpi=150, bbox_inches='tight')
    print("  -> plot_price_no_merge.png")
    plt.close()

    # Plot 2: 병합 있을 때
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(years, mean_yes, color='black', linewidth=2)
    ax.set_xlabel('경과 연수')
    ax.set_ylabel('토큰 가격 (원)')
    ax.set_title('토큰 병합 했을 때 평균 가격 경로')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0, T / 12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_price_with_merge.png', dpi=150, bbox_inches='tight')
    print("  -> plot_price_with_merge.png")
    plt.close()

    # Plot 3: 병합 발생 대표 경로 1개
    # 병합 횟수가 가장 많은(=대표적인) trial 선택
    merge_counts = [len(mh) for mh in res_yes['merge_hist']]
    if max(merge_counts) > 0:
        # 병합이 발생한 trial 중 병합 횟수가 중앙값인 것 선택
        trials_with_merge = [(i, c) for i, c in enumerate(merge_counts) if c > 0]
        trials_with_merge.sort(key=lambda x: x[1])
        best_idx = trials_with_merge[len(trials_with_merge) // 2][0]
    else:
        best_idx = 0

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(years, res_yes['price'][best_idx], color='steelblue', linewidth=1.5)

    # 병합 시점 표시
    for me in res_yes['merge_hist'][best_idx]:
        t_merge = me['t']
        ax.axvline(x=t_merge / 12, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.annotate(f"병합\n({me['price_before']:,.0f}→{me['price_after']:,.0f})",
                    xy=(t_merge / 12, me['price_after']),
                    xytext=(15, 20), textcoords='offset points',
                    fontsize=8, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    ax.set_xlabel('경과 연수')
    ax.set_ylabel('토큰 가격 (원)')
    ax.set_title('토큰 병합이 발생한 가격 경로 (대표 1개)')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0, T / 12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_price_merge_single.png', dpi=150, bbox_inches='tight')
    print(f"  -> plot_price_merge_single.png (trial #{best_idx}, merges={merge_counts[best_idx]})")
    plt.close()


# ================================================================== #
# Plot 4 & 5: 부도 발생 가격경로 + 하락률                             #
# ================================================================== #
def plot_default_scenarios():
    print("\n=== Plot 4&5: 부도 발생 가격경로 & 하락률 ===")
    T = T_SIMULATION
    N = 100

    # 부도가 발생한 trial을 찾기 위해 MC 실행
    print("  MC100 실행...")
    model = SolarPFModel(
        STATE_NAMES, X_NAMES, make_init_state(), T,
        seed=0, no_merge=True,
    )
    policy = SolarPFPolicy(model, policy_type='no_op')

    candidates = []  # (trial_idx, price_array, default_months)

    for trial in range(N):
        model_copy = copy(model)
        model_copy.reset(seed=trial)

        for _ in range(model_copy.T):
            decision = model_copy.build_decision({'no_op': 0})
            model_copy.step(decision)

        prices = np.array([h['state'].NAV_t / h['state'].token_count_t
                           for h in model_copy.history])

        # 부도 발생한 PF 탐지 (건설 중 부도)
        failed_pfs = [pf for pf in model_copy.pf_list if pf.status == 'FAILED']
        if failed_pfs:
            # 부도 시점 추정: PF의 start_month + local_time (부도 시 local_time에서 멈춤)
            default_months = []
            for pf in failed_pfs:
                default_t = pf.start_month + pf.local_time
                default_months.append(default_t)
            candidates.append((trial, prices, default_months))

    print(f"  부도 발생 trial: {len(candidates)}개")

    if len(candidates) < 3:
        print("  WARNING: 부도 발생 trial이 3개 미만, 추가 MC 실행...")
        for trial in range(N, N + 500):
            model_copy = copy(model)
            model_copy.reset(seed=trial)
            for _ in range(model_copy.T):
                decision = model_copy.build_decision({'no_op': 0})
                model_copy.step(decision)
            prices = np.array([h['state'].NAV_t / h['state'].token_count_t
                               for h in model_copy.history])
            failed_pfs = [pf for pf in model_copy.pf_list if pf.status == 'FAILED']
            if failed_pfs:
                default_months = [pf.start_month + pf.local_time for pf in failed_pfs]
                candidates.append((trial, prices, default_months))
            if len(candidates) >= 3:
                break

    # 3개 선택 (부도 시점이 다양하도록)
    candidates.sort(key=lambda x: x[2][0])  # 첫 부도 시점 기준 정렬
    step = max(1, len(candidates) // 3)
    selected = [candidates[i] for i in range(0, min(len(candidates), step * 3), step)][:3]

    years = np.arange(1, T + 1) / 12

    # Plot 4: 부도 발생 가격경로 3개
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    for idx, (trial_idx, prices, def_months) in enumerate(selected):
        ax = axes[idx]
        ax.plot(years, prices, color='steelblue', linewidth=1.2)
        for dm in def_months:
            if dm < T:
                ax.axvline(x=dm / 12, color='red', linestyle='--', alpha=0.8, linewidth=1.2)
                ax.annotate(f'PF 부도 (Month {dm})',
                            xy=(dm / 12, prices[min(dm, T - 1)]),
                            xytext=(15, 15), textcoords='offset points',
                            fontsize=9, color='red',
                            arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
        ax.set_ylabel('토큰 가격 (원)')
        ax.set_title(f'부도 발생 경로 #{idx + 1} (Trial {trial_idx})')
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlim(0, T / 12)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel('경과 연수')
    plt.tight_layout()
    plt.savefig('plot_default_price_paths.png', dpi=150, bbox_inches='tight')
    print("  -> plot_default_price_paths.png")
    plt.close()

    # Plot 5: 부도 발생 시 가격하락률 테이블
    # 모든 부도 이벤트를 모아서 하나의 테이블로 (가격 0인 행 제외)
    table_rows = []
    for idx, (trial_idx, prices, def_months) in enumerate(selected):
        for dm in def_months:
            if 1 < dm < T:
                price_before = prices[dm - 2]
                price_after = prices[dm]
                if price_before == 0 and price_after == 0:
                    continue  # 가격 0인 행 제외
                drop_amount = price_after - price_before
                drop_pct = drop_amount / price_before * 100 if price_before != 0 else 0
                table_rows.append([
                    f'{dm}개월 ({dm / 12:.1f}년)',
                    f'{price_before:,.0f}',
                    f'{price_after:,.0f}',
                    f'{drop_amount:,.0f}',
                    f'{drop_pct:+.2f}%',
                ])

    fig, ax = plt.subplots(figsize=(14, max(3, 1.5 + 0.6 * len(table_rows))))
    ax.axis('off')

    col_labels = ['부도 시점', '부도 전 가격', '부도 후 가격', '변동액', '변동률']
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # 헤더 스타일
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 데이터 행 스타일
    for i in range(1, len(table_rows) + 1):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor('#D9E2F3' if i % 2 == 1 else 'white')
        # 변동률 셀 빨간색
        table[i, 4].set_text_props(color='red', fontweight='bold')

    ax.set_title('부도 발생 시 토큰 가격 변동 상세', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('plot_default_drop_rates.png', dpi=150, bbox_inches='tight')
    print("  -> plot_default_drop_rates.png")
    plt.close()


# ================================================================== #
# Plot 6~8: 초기완공률 민감도 (3년)                                   #
# ================================================================== #
def plot_completion_sensitivity():
    print("\n=== Plot 6~8: 초기완공률 민감도 ===")
    T = 36  # 3년
    N = 100
    p_values = [0.99, 0.95, 0.90]
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    labels = ['P=0.99', 'P=0.95', 'P=0.90']

    results = {}
    for p in p_values:
        print(f"  P_COMPLETE={p} MC100...")
        res = run_mc(T, N, no_rollover=True, no_merge=True, p_complete=p)
        results[p] = np.mean(res['price'], axis=0)

    years = np.arange(1, T + 1) / 12

    # 개별 플롯 3개
    for i, p in enumerate(p_values):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(years, results[p], color=colors[i], linewidth=2, label=labels[i])
        ax.axvline(x=T_CONSTRUCTION / 12, color='red', linestyle='--', alpha=0.7,
                   linewidth=1.2, label='완공')
        ax.set_xlabel('경과 연수')
        ax.set_ylabel('토큰 가격 (원)')
        ax.set_title(f'초기완공률 {p} — 평균 가격 (0~3년)')
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(1/12))
        ax.set_xlim(0, T / 12)
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fname = f'plot_sensitivity_p{str(p).replace(".", "")}.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"  -> {fname}")
        plt.close()


# ================================================================== #
# 메인                                                                #
# ================================================================== #
if __name__ == '__main__':
    plot_merge_comparison()
    plot_default_scenarios()
    plot_completion_sensitivity()
    print("\n완료!")
