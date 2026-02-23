# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPRT (Solar PF Return Token) is a Monte Carlo simulation engine for modeling NAV (Net Asset Value) trajectories of tokenized solar project financing (PF) funds. The system simulates multiple solar PF projects with construction/operation phases, SMP/REC market dynamics, and token issuance/rollover mechanics.

## Commands

**Run simulation:**
```bash
python SolarPFDriver.py
```
This executes 50 Monte Carlo trials over 480 months (40 years) and saves results to `nav_simulation_result.png`.

## Architecture

### Core Files

| File | Role |
|------|------|
| `SolarPFModel.py` | Simulation environment: state management, exogenous shocks, PV engine, NAV calculation, transition functions |
| `SolarPFPolicy.py` | Policy classes (currently no-op) and episode runner |
| `SolarPFDriver.py` | MC simulation driver and visualization (entry point) |

### MDP Framework Structure

The model follows an MDP (Markov Decision Process) pattern:
- **State**: `(time_t, status_t, P_complete_t, PV_t, NAV_t, token_count_t)` - uses namedtuples for type safety
- **Status values**: `'PRE'` (construction), `'POST'` (operation), `'FAILED'` (default)
- **Transition**: `S_t × A_t × W_t → S_{t+1}` with monthly timesteps

### Key Components

1. **PFState dataclass** - Individual PF state tracking (status, completion probability, contracted price)
2. **Market buffers** - Rolling 6-month windows for SMP/REC prices and energy generation
3. **GBM price simulation** - SMP/REC follow geometric Brownian motion with caps/floors
4. **Completion probability** - `P(t) = h(t) × g(t)` hazard rate model
5. **Rollover mechanism** - 60-month cycle for new PF inclusion with token issuance constraints

### Key Parameters (in `SolarPFModel.py`)

- `T_CONSTRUCTION = 12` months, `T_OPERATION = 240` months
- `T_SIMULATION = 480` months (40 years with rollover)
- `T_ROLLOVER = 60` months (5-year rollover cycle)
- `INITIAL_PF_COUNT = 5`, `ROLLOVER_PF_COUNT = 5`
- `TOKEN_UNIT_PRICE = 10,000` (토큰 1개당 1만원)
- `TOKENS_PER_PF = 120,000` (PF당 토큰 수 = 12억 ÷ 1만)
- `P_DEFAULT = 0.003` (0.3%/month construction default)
- `P_COMPLETE_INIT = 0.95` (95% initial completion probability, based on ~2.5% real-world solar PF default rate)
- `R_DISCOUNT_ANNUAL = 0.045` (4.5% social discount rate)

### SMP/REC GBM Parameters

| 항목 | SMP | REC | 비고 |
|------|-----|-----|------|
| 초기가격 | 113 원/kWh | 77 원/kWh | 2025년 평균 |
| 드리프트 | +6%/년 | +6%/년 | 장기 +6.8% 반영 |
| 변동성 | 16%/년 | 14%/년 | 6개월 평균 효과로 절반 적용 |
| 하한 | 79 원 | 40 원 | 평균 - 1×STD |
| 상한 | 159 원 | 134 원 | 평균 + 1×STD |

**변동성 조정 근거**: 고정가격이 6개월 평균으로 산정되므로 σ/√6 ≈ 41% 수준으로 감소. 보수적으로 절반(50%) 적용.

### Evaluation Modes

```python
# Risk-neutral mode (default)
model = SolarPFModel(..., risk_adjust='NO')

# Real-world mode (with risk premium)
model = SolarPFModel(..., risk_adjust='YES')
```

## Design Documentation

- `nav_simulation_guide.md` - Detailed NAV simulation design specification
- `SMP_REC_발전량_DATA_ANALYSIS.md` - Data analysis and parameter estimation methodology
- `project_flow.md` - Business logic for STO token structure and rollover constraints
- `사용법.md` - Usage guide (Korean)

## Token Issuance Constraints

When adding new PFs via rollover, three conditions must be satisfied:
1. **Existing shareholder protection**: `b ≤ (y + ab) / (a + x)`
2. **Funding requirement**: `P × x = Q`
3. **New shareholder protection**: `P ≤ (y + ab) / (a + x)`

Where:
- `a` = existing tokens (초기 600,000개)
- `b` = current token price (= current NAV / a)
- `x` = new tokens to issue
- `y` = **inception value** of new PF (= completion value × P_COMPLETE_INIT)
- `P` = issue price per token (기준 1만원)
- `Q` = funding amount (= TOKEN_AMOUNT × ROLLOVER_PF_COUNT = 60억원)

**Important**: `y` is the inception value (완공후가치 × 0.95), not the completion value. This ensures consistency with NAV calculation where new PFs are valued at inception value during construction.

### Token Denomination

| 항목 | 값 |
|------|-----|
| 토큰 1개당 가격 | 10,000원 |
| PF당 토큰 수 | 120,000개 |
| 초기 발행 토큰 수 | 600,000개 (5 PF × 120,000) |
| 초기 자금조달 | 60억원 |
