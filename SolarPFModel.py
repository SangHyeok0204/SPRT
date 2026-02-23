"""
SolarPFModel.py — 태양광 PF NAV 시뮬레이션 환경

MDP BaseModel 패턴 (MDP_Framework_Structure.md)
도메인 명세 (nav_simulation_guide.md)
"""

import numpy as np
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# 하이퍼파라미터 (모듈 수준 상수)
# ============================================================

# 시간 구조 (월 단위)
T_CONSTRUCTION = 12                           # 1년
T_OPERATION = 240                             # 20년
T_SINGLE_PF = T_CONSTRUCTION + T_OPERATION    # 252개월
T_SIMULATION = 480                            # 40년 (롤오버)
T_ROLLOVER = 60                               # 롤오버 주기 (5년)

# 자본구조 (원 단위)
CAPITAL_TOTAL = 2_000_000_000                 # 프로젝트당 총 자금조달 20억
EQUITY_AMOUNT = 400_000_000                   # 자기자본 4억 (20%)
SENIOR_DEBT = 400_000_000                     # 선순위 대출 4억 (20%)
TOKEN_AMOUNT = 1_200_000_000                  # 토큰화 대상 12억 (60%)
INITIAL_PF_COUNT = 5                          # 초기 운영 PF 수
ROLLOVER_PF_COUNT = 5                         # 롤오버 시 편입 PF 수

# 토큰 단위 (소액 투자자 접근성)
TOKEN_UNIT_PRICE = 10_000                     # 토큰 1개당 가격 1만원
TOKENS_PER_PF = TOKEN_AMOUNT // TOKEN_UNIT_PRICE  # PF당 토큰 수 = 12억/1만 = 120,000개

# 수익 분배 구조
EQUITY_SHARE = 0.25                           # 지분수익 비율 (1/4)
TOKEN_SHARE = 0.75                            # 토큰수익 비율 (3/4)
# 순현금흐름 = 지분수익 + 토큰수익 (1:3 비율)

# 외생 이벤트 확률 (월별)
P_DEFAULT = 0.003         # 0.3%/월 — 건설 디폴트 (일별 0.01% 환산)
P_CURTAIL = 0.003         # 0.3%/월 — 운영 출력제한 (일별 0.01% 환산)

# 재무 파라미터
R_DISCOUNT_ANNUAL = 0.045
R_MONTHLY = (1 + R_DISCOUNT_ANNUAL) ** (1 / 12) - 1

# FCF 비용구조 (플레이스홀더 — 추후 조정)
# 총비용 = 대출금 원금상환 + 선순위 이자비용 + OPEX + 재무비 + 세금
SENIOR_INTEREST_RATE = 0.05                   # 선순위 대출 연이자율 5%
OPEX_ANNUAL = 20_000_000                      # 연간 OPEX 2천만원 (1MW 기준)
OPEX_MONTHLY = OPEX_ANNUAL / 12
TAX_RATE = 0.22                               # 법인세율 22%
DEBT_REPAYMENT_YEARS = 15                     # 대출금 상환기간 15년
DEBT_MONTHLY_PAYMENT = SENIOR_DEBT / (DEBT_REPAYMENT_YEARS * 12)  # 균등상환

# 시장 데이터 윈도우 및 장기계약
WINDOW_SIZE = 6                               # 고정가격 산정 윈도우 (6개월)
# 장기계약: 편입 시점 6개월 평균 (SMP + 1×REC) 가격으로 운영기간 전체 수익 고정

# 완공 모델 (위험률 기반)
LAMBDA_DEFAULT = 0.15    # 월별 위험률 (일별 0.005 × 30 환산)
P_COMPLETE_INIT = 0.95   # 초기 완공 확률 95% (신재생에너지 PF 부도율 ~2.5% 반영)

# SMP-REC 상관: 월별 로그수익률 ρ = -0.03 (통계적 비유의, 독립 처리)

# 실데이터 기반 GBM 파라미터 (SMP_REC_발전량_DATA_ANALYSIS.md 참조)
# 변동성 조정: 고정가격이 6개월 평균으로 산정되므로 이론적으로 σ/√6 ≈ 0.41배
# 보수적으로 절반(0.5배) 적용 → SMP 16%, REC 14%
SMP_PARAMS = dict(
    initial_price=113.0,      # 원/kWh (2025 EPSIS 평균)
    annual_drift=0.06,        # +6%/년 (장기 +6.8% 반영)
    annual_vol=0.16,          # 16% (32.4% × 0.5, 6개월 평균 효과 보수적 적용)
    initial_volume=1000.0,    # MWh (미사용, 향후 확장용)
)
REC_PARAMS = dict(
    initial_price=77.0,       # 원/kWh (77,000원/REC ÷ 1000 kWh/MWh)
    annual_drift=0.06,        # +6%/년 (SMP와 동일하게 조정)
    annual_vol=0.14,          # 14% (28.1% × 0.5, 6개월 평균 효과 보수적 적용)
    initial_volume=500.0,     # MWh (미사용, 향후 확장용)
)

# SMP/REC 가격 상한/하한 (40년 GBM 발산 방지)
# 변동성 축소(16%, 14%)로 발산 위험 감소 → 범위를 평균 ± 1×STD로 확대
SMP_FLOOR = 79.0              # SMP 하한 (118.99 - 40.01)
SMP_CAP = 159.0               # SMP 상한 (118.99 + 40.01)
REC_FLOOR = 40.0              # REC 하한 (86.99 - 46.98)
REC_CAP = 134.0               # REC 상한 (86.99 + 46.98)

# 발전량 파라미터 (1MW 태양광 PF, 이용률 14.5% 가정)
# 월별 계절성 계수: 2024년 한국전력거래소 일별 발전량 실데이터 기반
# 계수 = 해당월 일평균 / 연 일평균 (1.0 = 연평균 수준)
MONTHLY_SEASONAL = [
    0.698, 0.657, 1.092, 1.138, 1.417, 1.284,   # 1~6월
    0.954, 1.256, 1.083, 0.850, 0.809, 0.749,   # 7~12월
]
# 월별 계절계수는 MONTHLY_SEASONAL을 직접 사용 (인덱스 = t % 12)

ENERGY_PARAMS = dict(
    base_generation=105.9,    # MWh/월 (1MW × 24h × 30.4d × 14.5%)
    monthly_vol=0.005,        # 발전량 변동성 축소 (0.5%)
)


# ============================================================
# 개별 PF 상태 클래스
# ============================================================

@dataclass
class PFState:
    """개별 태양광 PF의 상태."""
    pf_id: int                                # PF 식별자
    start_month: int                          # 편입 시점 (전역 시간 기준)
    status: str = 'PRE'                       # 'PRE', 'POST', 'FAILED'
    P_complete: float = P_COMPLETE_INIT       # 완공 확률
    contracted_price: Optional[float] = None  # 장기계약 고정가격
    local_time: int = 0                       # PF 내부 시간 (편입 후 경과 월)

    def is_active(self):
        """PF가 활성 상태인지 (FAILED 아님, 운영기간 내)."""
        return self.status != 'FAILED' and self.local_time < T_SINGLE_PF

    def is_expired(self):
        """PF 운영기간 종료 여부."""
        return self.local_time >= T_SINGLE_PF


# ============================================================
# 모델 클래스
# ============================================================

class SolarPFModel:
    """태양광 PF NAV 시뮬레이션 환경 (MDP BaseModel 패턴)."""

    def __init__(self, state_names, x_names, s_0, T,
                 risk_adjust='NO', seed=42):
        """
        매개변수
        ----------
        state_names : list[str]   — 예: ['time_t','status_t','P_complete_t','PV_t','NAV_t']
        x_names     : list[str]   — 예: ['no_op']
        s_0         : dict        — 초기 상태 값
        T           : int         — 에피소드 길이 (일)
        risk_adjust : str         — 'NO' (위험중립) 또는 'YES' (현실세계)
        seed        : int         — 난수 시드
        """
        # namedtuple 동적 생성 (MDP 패턴 1)
        self.State = namedtuple('State', state_names)
        self.Decision = namedtuple('Decision', x_names)

        self.state_names = state_names
        self.x_names = x_names
        self.init_state = dict(s_0)
        self.T = T
        self.risk_adjust = risk_adjust
        self.seed = seed
        self.prng = np.random.RandomState(seed)

        # 내부 시장 데이터 버퍼 (메모리 효율을 위해 State 외부에 보관)
        self.SMP_buffer = None
        self.REC_buffer = None
        self.energy_buffer = None

        # 장기계약 고정가격 (편입 시 6개월 평균 SMP+REC) — 단일 PF용 (하위호환)
        self.contracted_price = None

        # 다중 PF 관리
        self.pf_list = []
        self.pf_id_counter = 0
        self.rollover_schedule = []
        self.token_count = 0.0                # 총 발행 토큰 수
        self.token_price = 0.0                # 토큰 가격

        # 에피소드 관리 변수
        self.t = 0
        self.obj = 0.0
        self._init_market_buffers()
        self._init_pf_list()
        self.state = self.build_state(self.init_state)
        self.history = []

    # ------------------------------------------------------------------ #
    #  어댑터                                                             #
    # ------------------------------------------------------------------ #

    def build_state(self, info):
        """dict → State namedtuple 변환"""
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        """dict → Decision namedtuple 변환"""
        return self.Decision(*[info[k] for k in self.x_names])

    def reset(self, seed=None):
        """에피소드를 초기 상태로 리셋."""
        self.t = 0
        self.obj = 0.0
        if seed is not None:
            self.seed = seed
        self.prng = np.random.RandomState(self.seed)
        self._init_market_buffers()
        self.contracted_price = None
        self.pf_id_counter = 0
        self._init_pf_list()
        self.state = self.build_state(self.init_state)
        self.history = []

    # ------------------------------------------------------------------ #
    #  시장 데이터 버퍼 (Task #4)                                         #
    # ------------------------------------------------------------------ #

    def _init_pf_list(self, count=INITIAL_PF_COUNT):
        """초기 PF 목록 생성."""
        self.pf_list = []
        for _ in range(count):
            pf = PFState(
                pf_id=self.pf_id_counter,
                start_month=0,
                status='PRE',
                P_complete=P_COMPLETE_INIT,
            )
            self.pf_list.append(pf)
            self.pf_id_counter += 1
        # 초기 토큰 발행: PF당 120,000개 × 5개 PF = 600,000개
        self.token_count = float(count * TOKENS_PER_PF)
        self.token_price = TOKEN_UNIT_PRICE   # 토큰 1개당 가격 = 1만원

    def _init_market_buffers(self):
        """초기 합성 시장 데이터 윈도우 생성."""
        self.SMP_buffer = np.column_stack([
            np.full(WINDOW_SIZE, SMP_PARAMS['initial_price']),
            np.full(WINDOW_SIZE, SMP_PARAMS['initial_volume']),
        ])
        self.REC_buffer = np.column_stack([
            np.full(WINDOW_SIZE, REC_PARAMS['initial_price']),
            np.full(WINDOW_SIZE, REC_PARAMS['initial_volume']),
        ])
        self.energy_buffer = np.full(WINDOW_SIZE, ENERGY_PARAMS['base_generation'])

    def _calculate_contracted_price(self):
        """편입 시점 6개월 평균 (SMP + 1×REC) 고정가격 산정."""
        avg_smp = np.mean(self.SMP_buffer[:, 0])
        avg_rec = np.mean(self.REC_buffer[:, 0])
        return avg_smp + avg_rec

    def _generate_market_step(self):
        """합성 시장 데이터를 한 달 전진시킴 (상관 GBM + 실데이터 계절성)."""
        dt = 1 / 12

        # 독립 브라운 운동 (SMP-REC 상관 ρ≈0, 독립 처리)
        z_smp = self.prng.randn()
        z_rec = self.prng.randn()

        # SMP 가격 — GBM + 상한/하한 클리핑
        last_smp = self.SMP_buffer[-1, 0]
        mu_s, sig_s = SMP_PARAMS['annual_drift'], SMP_PARAMS['annual_vol']
        smp_price = last_smp * np.exp(
            (mu_s - 0.5 * sig_s**2) * dt
            + sig_s * np.sqrt(dt) * z_smp
        )
        smp_price = np.clip(smp_price, SMP_FLOOR, SMP_CAP)  # 상한/하한 적용
        smp_volume = max(0.0, SMP_PARAMS['initial_volume']
                         * (1 + 0.1 * self.prng.randn()))

        # REC 가격 — GBM + 상한/하한 클리핑 (SMP와 독립)
        last_rec = self.REC_buffer[-1, 0]
        mu_r, sig_r = REC_PARAMS['annual_drift'], REC_PARAMS['annual_vol']
        rec_price = last_rec * np.exp(
            (mu_r - 0.5 * sig_r**2) * dt
            + sig_r * np.sqrt(dt) * z_rec
        )
        rec_price = np.clip(rec_price, REC_FLOOR, REC_CAP)  # 상한/하한 적용
        rec_volume = max(0.0, REC_PARAMS['initial_volume']
                         * (1 + 0.1 * self.prng.randn()))

        # 발전량 — 월별 계절계수 + 노이즈
        month_of_year = self.t % 12
        seasonal = MONTHLY_SEASONAL[month_of_year]
        energy = max(0.0,
                     ENERGY_PARAMS['base_generation'] * seasonal
                     * (1 + ENERGY_PARAMS['monthly_vol'] * self.prng.randn()))

        # 윈도우 인플레이스 갱신
        self.SMP_buffer[:-1] = self.SMP_buffer[1:]
        self.SMP_buffer[-1] = [smp_price, smp_volume]

        self.REC_buffer[:-1] = self.REC_buffer[1:]
        self.REC_buffer[-1] = [rec_price, rec_volume]

        self.energy_buffer[:-1] = self.energy_buffer[1:]
        self.energy_buffer[-1] = energy

    # ------------------------------------------------------------------ #
    #  외생 이벤트 (Task #2)                                              #
    # ------------------------------------------------------------------ #

    def exog_info_fn(self, decision):
        """월별 베르누이 외생 이벤트 샘플링.

        - 건설 디폴트: completion_probability_fn의 g(t)에서 처리
        - 운영 출력제한: 여기서 처리 (POST 상태에서만)
        """
        status = self.state.status_t
        curtail_today = False

        if status == 'POST':
            curtail_today = (self.prng.random() < P_CURTAIL)

        return {'curtail_today': curtail_today}

    # ------------------------------------------------------------------ #
    #  완공 확률 (Task #3) — P(t) = h(t) × g(t) 구조                      #
    # ------------------------------------------------------------------ #

    def _sample_g_t(self):
        """g(t): 매월 0.3% 확률로 0이 되는 바이너리 변수.

        g(t) = 0 발생 시 프로젝트 실패 (FAILED).
        """
        return 0 if self.prng.random() < P_DEFAULT else 1

    def _h_t(self, P_prev, lambda_t=LAMBDA_DEFAULT):
        """h(t): hazard rate 기반 완공 확률 증가.

        h_t = 1 - (1 - h_{t-1}) * exp(-lambda_t)
        """
        return 1.0 - (1.0 - P_prev) * np.exp(-lambda_t)

    def completion_probability_fn(self, P_prev, lambda_t=LAMBDA_DEFAULT):
        """P(t) = h(t) × g(t).

        - h(t): hazard rate 기반 완공 확률 (연속 증가)
        - g(t): 바이너리 (0.3% 확률로 0, 이 경우 FAILED)

        Returns
        -------
        tuple: (P_t, g_t) — P_t는 완공확률, g_t는 바이너리 (0이면 실패)
        """
        g_t = self._sample_g_t()
        if g_t == 0:
            return (0.0, 0)  # 프로젝트 실패
        h_t = self._h_t(P_prev, lambda_t)
        return (h_t, 1) 

    # ------------------------------------------------------------------ #
    #  PV 엔진 — 단순 DCF 플레이스홀더 (Task #5)                         #
    # ------------------------------------------------------------------ #

    def pv_engine(self, status, time_t, curtail_today=False):
        """미래 운영 현금흐름의 현재가치.

        FCF = Revenue - OPEX - Taxes - 이자비용 - 원금상환
        - Revenue: 장기계약 고정가격 × 발전량 (POST) 또는 시장 윈도우 평균 (PRE)
        - 토큰 보유자에게 귀속되는 부분만 계산 (TOKEN_SHARE)
        """
        if status == 'FAILED':
            return 0.0

        # 발전량 추정
        avg_energy = np.mean(self.energy_buffer)

        # 수익 계산 (원/kWh × MWh × 1000 = 원)
        if status == 'POST' and self.contracted_price is not None:
            # 운영기간: 장기계약 고정가격 사용
            price = self.contracted_price
        else:
            # 건설기간: 시장 윈도우 평균으로 추정
            avg_smp = np.mean(self.SMP_buffer[:, 0])
            avg_rec = np.mean(self.REC_buffer[:, 0])
            price = avg_smp + avg_rec

        monthly_revenue = price * avg_energy * 1000

        # 비용 계산
        monthly_interest = SENIOR_DEBT * (SENIOR_INTEREST_RATE / 12)
        # 상환기간 내에만 원금상환 발생
        op_month = max(0, time_t - T_CONSTRUCTION)
        if op_month < DEBT_REPAYMENT_YEARS * 12:
            monthly_debt = DEBT_MONTHLY_PAYMENT
        else:
            monthly_debt = 0.0

        # FCF = Revenue - OPEX - 이자 - 원금상환
        gross_cf = monthly_revenue - OPEX_MONTHLY - monthly_interest - monthly_debt
        # 세후 현금흐름
        if gross_cf > 0:
            monthly_fcf = gross_cf * (1 - TAX_RATE)
        else:
            monthly_fcf = gross_cf  # 손실 시 세금 없음

        # 토큰 보유자 귀속분
        token_cf = monthly_fcf * TOKEN_SHARE

        # 잔여 기간 계산
        if status == 'PRE':
            months_to_start = max(0, T_CONSTRUCTION - time_t)
            remaining_op = T_OPERATION
        else:   # POST
            months_to_start = 0
            remaining_op = max(0, T_SINGLE_PF - time_t)

        if remaining_op <= 0:
            return 0.0

        # PV = 월별FCF × 연금계수(잔여운영월) × 할인계수(운영시작까지)
        annuity = (1 - (1 + R_MONTHLY) ** (-remaining_op)) / R_MONTHLY
        discount = (1 + R_MONTHLY) ** (-months_to_start)
        pv = token_cf * annuity * discount

        # 출력제한: 해당월 운영 현금흐름 = 0
        if curtail_today and status == 'POST':
            pv -= token_cf

        return pv

    # ------------------------------------------------------------------ #
    #  개별 PF 업데이트 (다중 PF 구조)                                    #
    # ------------------------------------------------------------------ #

    def _update_single_pf(self, pf: PFState, curtail_today: bool):
        """개별 PF 상태를 한 달 전진시키고 PV 반환."""
        if not pf.is_active():
            return 0.0

        # 로컬 시간 증가
        pf.local_time += 1

        # PRE 상태: 완공 확률 및 g(t) 샘플링
        if pf.status == 'PRE':
            P_t, g_t = self.completion_probability_fn(pf.P_complete)
            pf.P_complete = P_t
            if g_t == 0:
                pf.status = 'FAILED'
                return 0.0
            elif pf.local_time >= T_CONSTRUCTION:
                pf.status = 'POST'
                # 장기계약 고정가격 산정
                pf.contracted_price = self._calculate_contracted_price()

        # PV 계산
        pv = self._pv_engine_for_pf(pf, curtail_today)

        # NAV 계산 (PRE 상태면 P_complete 반영)
        if pf.status == 'PRE':
            return pf.P_complete * pv
        else:
            return pv

    def _pv_engine_for_pf(self, pf: PFState, curtail_today: bool):
        """개별 PF의 PV 계산."""
        if pf.status == 'FAILED':
            return 0.0

        avg_energy = np.mean(self.energy_buffer)

        # 수익 계산
        if pf.status == 'POST' and pf.contracted_price is not None:
            price = pf.contracted_price
        else:
            avg_smp = np.mean(self.SMP_buffer[:, 0])
            avg_rec = np.mean(self.REC_buffer[:, 0])
            price = avg_smp + avg_rec

        monthly_revenue = price * avg_energy * 1000

        # 비용 계산
        monthly_interest = SENIOR_DEBT * (SENIOR_INTEREST_RATE / 12)
        op_month = max(0, pf.local_time - T_CONSTRUCTION)
        if op_month < DEBT_REPAYMENT_YEARS * 12:
            monthly_debt = DEBT_MONTHLY_PAYMENT
        else:
            monthly_debt = 0.0

        gross_cf = monthly_revenue - OPEX_MONTHLY - monthly_interest - monthly_debt
        if gross_cf > 0:
            monthly_fcf = gross_cf * (1 - TAX_RATE)
        else:
            monthly_fcf = gross_cf

        token_cf = monthly_fcf * TOKEN_SHARE

        # 잔여 기간
        if pf.status == 'PRE':
            months_to_start = max(0, T_CONSTRUCTION - pf.local_time)
            remaining_op = T_OPERATION
        else:
            months_to_start = 0
            remaining_op = max(0, T_SINGLE_PF - pf.local_time)

        if remaining_op <= 0:
            return 0.0

        annuity = (1 - (1 + R_MONTHLY) ** (-remaining_op)) / R_MONTHLY
        discount = (1 + R_MONTHLY) ** (-months_to_start)
        pv = token_cf * annuity * discount

        if curtail_today and pf.status == 'POST':
            pv -= token_cf

        return pv

    def _calculate_monthly_distribution_for_pf(self, pf: PFState, curtail_today: bool):
        """개별 PF의 월별 분배금 계산 (POST 상태에서만 분배).

        분배금 = (매출액 - 총비용) × (1 - 세율) × 토큰비율
        건설 중(PRE)에는 분배금 = 0
        """
        # PRE 또는 FAILED 상태면 분배 없음
        if pf.status != 'POST':
            return 0.0

        # 출력제한 발생 시 해당월 수익 = 0
        if curtail_today:
            return 0.0

        # 운영기간 종료 체크
        if pf.local_time >= T_SINGLE_PF:
            return 0.0

        avg_energy = np.mean(self.energy_buffer)

        # 수익 계산 (장기계약 고정가격 사용)
        if pf.contracted_price is not None:
            price = pf.contracted_price
        else:
            avg_smp = np.mean(self.SMP_buffer[:, 0])
            avg_rec = np.mean(self.REC_buffer[:, 0])
            price = avg_smp + avg_rec

        monthly_revenue = price * avg_energy * 1000

        # 비용 계산
        monthly_interest = SENIOR_DEBT * (SENIOR_INTEREST_RATE / 12)
        op_month = max(0, pf.local_time - T_CONSTRUCTION)
        if op_month < DEBT_REPAYMENT_YEARS * 12:
            monthly_debt = DEBT_MONTHLY_PAYMENT
        else:
            monthly_debt = 0.0

        gross_cf = monthly_revenue - OPEX_MONTHLY - monthly_interest - monthly_debt

        if gross_cf > 0:
            monthly_fcf = gross_cf * (1 - TAX_RATE)
        else:
            monthly_fcf = gross_cf  # 손실 시 세금 없음

        # 토큰 보유자 귀속분 (75%)
        token_distribution = monthly_fcf * TOKEN_SHARE

        return max(0.0, token_distribution)  # 음수 분배 없음

    def _calculate_total_nav(self, curtail_today: bool):
        """모든 활성 PF의 NAV 합산."""
        total_nav = 0.0
        for pf in self.pf_list:
            if pf.is_active():
                nav = self._update_single_pf(pf, curtail_today)
                total_nav += nav
        return total_nav

    # ------------------------------------------------------------------ #
    #  NAV 계산기 (Task #6)                                               #
    # ------------------------------------------------------------------ #

    def nav_calculator_rn(self, P_t, PV_t):
        """위험중립: NAV = P_t * PV_t"""
        return P_t * PV_t

    def nav_calculator_rw(self, P_t, PV_t, risk_premium=0.02):
        """현실세계: NAV = P_t * PV_t * (1 - risk_premium)  [플레이스홀더]"""
        return P_t * PV_t * (1.0 - risk_premium)

    # ------------------------------------------------------------------ #
    #  전이 함수 (Task #7) — 다중 PF 구조                                 #
    # ------------------------------------------------------------------ #

    def transition_fn(self, decision, exog_info):
        """S_t x A_t x W_t → S_{t+1} (다중 PF 구조)."""
        time_t = self.t
        curtail_today = exog_info['curtail_today']

        # 1. 시장 데이터 전진 (매월 업데이트)
        self._generate_market_step()

        # 2. 모든 PF 업데이트 및 NAV 합산
        total_nav = 0.0
        total_pv = 0.0
        total_distribution = 0.0  # 월별 분배금 합산
        active_count = 0
        avg_p_complete = 0.0

        for pf in self.pf_list:
            if pf.is_active():
                nav = self._update_single_pf(pf, curtail_today)
                total_nav += nav
                total_pv += self._pv_engine_for_pf(pf, curtail_today)
                # POST 상태 PF만 분배금 발생
                total_distribution += self._calculate_monthly_distribution_for_pf(pf, curtail_today)
                active_count += 1
                if pf.status == 'PRE':
                    avg_p_complete += pf.P_complete

        # 현실세계 조정
        if self.risk_adjust == 'YES':
            total_nav *= (1.0 - 0.02)  # 리스크 프리미엄 2%
            total_pv *= (1.0 - 0.02)

        # 3. 롤오버 훅 (신규 PF 편입 시 NAV에 반영)
        rollover_nav = self.apply_events(time_t, total_nav)
        if rollover_nav is not None:
            total_nav = rollover_nav
            # 신규 PF들의 PV도 추가
            for pf in self.pf_list:
                if pf.is_active() and pf.start_month == time_t:
                    total_pv += self._pv_engine_for_pf(pf, curtail_today)
                    active_count += 1

        # 4. 평균 완공 확률 (PRE 상태 PF만) — 롤오버 후 재계산
        pre_count = sum(1 for pf in self.pf_list if pf.status == 'PRE' and pf.is_active())
        if pre_count > 0:
            avg_p_complete = sum(pf.P_complete for pf in self.pf_list
                                 if pf.status == 'PRE' and pf.is_active()) / pre_count
        else:
            avg_p_complete = 1.0

        # 5. 펀드 전체 상태 결정
        if active_count == 0:
            fund_status = 'FAILED'
        elif all(pf.status == 'POST' for pf in self.pf_list if pf.is_active()):
            fund_status = 'POST'
        else:
            fund_status = 'PRE'

        return {
            'time_t': time_t,
            'status_t': fund_status,
            'P_complete_t': avg_p_complete,
            'PV_t': total_pv,
            'NAV_t': total_nav,
            'token_count_t': self.token_count,
            'monthly_distribution_t': total_distribution,  # 월별 분배금 (펀드 전체)
        }

    # ------------------------------------------------------------------ #
    #  목적함수 (보상)                                                     #
    # ------------------------------------------------------------------ #

    def objective_fn(self, decision, exog_info):
        """현재 NAV를 보상으로 반환 (최적화 없음, 기록 전용)."""
        return self.state.NAV_t

    # ------------------------------------------------------------------ #
    #  스텝 — 1일 타임스텝 (MDP 패턴 2)                                   #
    # ------------------------------------------------------------------ #

    def step(self, decision):
        """1일 타임스텝 실행: 외생충격 → 전이 → 기록."""
        self.t += 1

        # 1. 외생 정보
        exog_info = self.exog_info_fn(decision)

        # 2. 상태 전이 (파생 변수 계산 포함)
        new_state_dict = self.transition_fn(decision, exog_info)
        self.state = self.build_state(new_state_dict)

        # 3. 보상
        reward = self.state.NAV_t
        self.obj = reward

        # 4. 이력 기록
        self.history.append({
            't': self.t,
            'state': self.state,
            'decision': decision,
            'exog_info': exog_info,
            'reward': reward,
        })

    # ------------------------------------------------------------------ #
    #  롤오버 훅 (Task #10) — 60개월 주기 롤오버                          #
    # ------------------------------------------------------------------ #

    def apply_events(self, t, current_nav_before_rollover=None):
        """PF 롤오버 로직.

        60개월마다 신규 5개 PF 편입.
        토큰 증자 3가지 조건 검증 후 실행.

        Parameters
        ----------
        t : int
            현재 시점 (월)
        current_nav_before_rollover : float, optional
            롤오버 전 NAV (transition_fn에서 전달)

        Returns
        -------
        float or None
            롤오버 성공 시 신규 NAV (기존 NAV + 신규 PF 가치), 실패 시 None
        """
        # 롤오버 시점 체크 (60개월마다, t=0 제외)
        if t == 0 or t % T_ROLLOVER != 0:
            return None

        # 만료된 PF 제거
        self.pf_list = [pf for pf in self.pf_list if not pf.is_expired()]

        # 현재 펀드 가치 계산 (기존 토큰 총 가치)
        # transition_fn에서 전달받은 값 사용 (risk_adjust 적용된 값)
        if current_nav_before_rollover is not None:
            current_nav = current_nav_before_rollover
        else:
            current_nav = sum(
                self._pv_engine_for_pf(pf, False)
                for pf in self.pf_list if pf.is_active()
            )

        # 신규 PF 가치 (5개 PF의 예상 가치)
        new_pf_value = self._estimate_new_pf_value()

        # 자금조달액 (Q)
        Q = TOKEN_AMOUNT * ROLLOVER_PF_COUNT

        # 토큰 증자 조건 검증 및 실행
        result = self._execute_token_issuance(current_nav, new_pf_value, Q)

        if result['success']:
            # 신규 PF 편입
            for _ in range(ROLLOVER_PF_COUNT):
                pf = PFState(
                    pf_id=self.pf_id_counter,
                    start_month=t,
                    status='PRE',
                    P_complete=P_COMPLETE_INIT,
                )
                self.pf_list.append(pf)
                self.pf_id_counter += 1

            # 토큰 수 업데이트
            self.token_count += result['new_tokens']

            # 롤오버 기록
            self.rollover_schedule.append({
                't': t,
                'new_pf_count': ROLLOVER_PF_COUNT,
                'new_tokens': result['new_tokens'],
                'token_price': result['new_price'],
                'issue_price': result['issue_price'],
            })

            # 신규 NAV 반환 (기존 NAV + 신규 PF 가치)
            # 신규 PF는 PRE 상태이므로 P_COMPLETE_INIT을 곱해야 함
            new_pf_nav = new_pf_value * P_COMPLETE_INIT
            return current_nav + new_pf_nav

        return None

    def _estimate_new_pf_value(self):
        """신규 PF 5개의 예상 가치 (편입 시점 기준)."""
        # 현재 시장 가격 기준으로 추정
        avg_smp = np.mean(self.SMP_buffer[:, 0])
        avg_rec = np.mean(self.REC_buffer[:, 0])
        avg_energy = np.mean(self.energy_buffer)

        # 월별 예상 수익
        monthly_revenue = (avg_smp + avg_rec) * avg_energy * 1000
        monthly_interest = SENIOR_DEBT * (SENIOR_INTEREST_RATE / 12)
        gross_cf = monthly_revenue - OPEX_MONTHLY - monthly_interest - DEBT_MONTHLY_PAYMENT
        if gross_cf > 0:
            monthly_fcf = gross_cf * (1 - TAX_RATE)
        else:
            monthly_fcf = gross_cf
        token_cf = monthly_fcf * TOKEN_SHARE

        # PV 계산 (운영기간 전체)
        annuity = (1 - (1 + R_MONTHLY) ** (-T_OPERATION)) / R_MONTHLY
        discount = (1 + R_MONTHLY) ** (-T_CONSTRUCTION)
        single_pf_value = token_cf * annuity * discount

        return single_pf_value * ROLLOVER_PF_COUNT

    def _sample_winning_bid(self, x_min, x_max):
        """낙찰 확률밀도함수 기반 토큰 발행 수 샘플링.

        p(x) ∝ exp(b * (x - x_min) / (x_max - x_min))
        - x_max (토큰 많음, 신규투자자 유리): 낙찰확률 높음 (80%)
        - x_min (토큰 적음, 기존투자자 유리): 낙찰확률 낮음 (거의 0%)

        현실적으로 낙찰 경쟁에서 신규투자자가 더 많은 토큰을 받으려 할수록
        낙찰 가능성이 높아지는 구조.
        """
        if x_max <= x_min:
            return x_min

        # 지수 분포 파라미터 (0에서 1로 정규화된 범위에서)
        # b = ln(0.8/0.01) ≈ 4.38 (x_max에서 80%, x_min에서 ~1%)
        b = 4.0

        # 역변환 샘플링
        u = self.prng.random()
        # CDF: F(t) = (exp(bt) - 1) / (exp(b) - 1), t ∈ [0, 1]
        # 역함수: t = ln(1 + u*(exp(b) - 1)) / b
        exp_b = np.exp(b)
        t = np.log(1 + u * (exp_b - 1)) / b

        # t를 [x_min, x_max] 범위로 변환
        x = x_min + t * (x_max - x_min)
        return x

    def _execute_token_issuance(self, current_nav, new_pf_value, Q):
        """토큰 증자 3가지 조건 검증 및 실행 (편입 시점 가치 기준).

        조건 1: b ≤ (y' + current_nav) / (a + x) — 기존 주주 보호
        조건 2: P × x = Q — 자금조달
        조건 3: P ≤ (y' + current_nav) / (a + x) — 신규 주주 보호

        여기서:
        - a = 기존 토큰 수
        - b = 실제 토큰 가격 = current_nav / a
        - y' = 신규 PF 편입 시점 가치 = 완공후가치 × P_COMPLETE_INIT
        - x = 신규 발행 토큰 수
        - P = 토큰 공모가
        - Q = 자금조달액

        Returns
        -------
        dict: {success, new_tokens, issue_price, new_price}
        """
        a = self.token_count           # 기존 토큰 수
        # 신규 PF 편입 시점 가치 (완공후가치 × 초기완공확률)
        y = new_pf_value * P_COMPLETE_INIT

        if a == 0 or current_nav <= 0:
            # 초기 상태 — 단순 발행 (PF당 120,000토큰 × 5 PF)
            new_tokens = ROLLOVER_PF_COUNT * TOKENS_PER_PF
            return {
                'success': True,
                'new_tokens': new_tokens,
                'issue_price': TOKEN_UNIT_PRICE,
                'new_price': TOKEN_UNIT_PRICE,
            }

        # 실제 토큰 가격 (current_nav 기준)
        b = current_nav / a

        # 조건을 만족하는 x 범위 계산
        # 조건 1: b ≤ (y + current_nav) / (a + x)
        #         b(a + x) ≤ y + current_nav
        #         ba + bx ≤ y + current_nav
        #         bx ≤ y  (∵ ba = current_nav)
        #         x ≤ y / b
        #
        # 조건 3: P ≤ (y + current_nav) / (a + x), P = Q / x
        #         Q / x ≤ (y + current_nav) / (a + x)
        #         Q(a + x) ≤ x(y + current_nav)
        #         Qa ≤ x(y + current_nav - Q)
        #         x ≥ Qa / (y + current_nav - Q)  (if y + current_nav > Q)

        total_value = y + current_nav
        if total_value <= Q:
            # 신규 가치 + 기존 가치가 자금조달액보다 작음 — 증자 불가
            return {'success': False, 'new_tokens': 0, 'issue_price': 0, 'new_price': b}

        x_min = Q * a / (total_value - Q)
        x_max = y / b

        if x_min >= x_max:
            # 조건 만족 범위 없음
            return {'success': False, 'new_tokens': 0, 'issue_price': 0, 'new_price': b}

        # 낙찰 확률밀도함수 기반 x 샘플링
        x = self._sample_winning_bid(x_min, x_max)
        P = Q / x                       # 토큰 공모가
        new_price = total_value / (a + x)

        # 토큰 가격 업데이트 (실제 NAV 기준)
        self.token_price = new_price

        return {
            'success': True,
            'new_tokens': x,
            'issue_price': P,
            'new_price': new_price,
        }
