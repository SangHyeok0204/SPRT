"""
SolarPFModel.py — 태양광 PF NAV 시뮬레이션 환경

MDP BaseModel 패턴 (MDP_Framework_Structure.md)
도메인 명세 (nav_simulation_guide.md)
"""

import numpy as np
from collections import namedtuple

# ============================================================
# 하이퍼파라미터 (모듈 수준 상수)
# ============================================================

# 시간 구조 (월 단위)
T_CONSTRUCTION = 24                           # 2년
T_OPERATION = 240                             # 20년
T_SINGLE_PF = T_CONSTRUCTION + T_OPERATION    # 264개월
T_SIMULATION = 480                            # 40년 (롤오버)

# 외생 이벤트 확률 (월별)
P_DEFAULT = 0.003         # 0.3%/월 — 건설 디폴트 (일별 0.01% 환산)
P_CURTAIL = 0.003         # 0.3%/월 — 운영 출력제한 (일별 0.01% 환산)

# 재무 파라미터
R_DISCOUNT_ANNUAL = 0.045
R_MONTHLY = (1 + R_DISCOUNT_ANNUAL) ** (1 / 12) - 1

# 시장 데이터 윈도우
WINDOW_SIZE = 6

# 완공 모델 (위험률 기반)
LAMBDA_DEFAULT = 0.15    # 월별 위험률 (일별 0.005 × 30 환산)
P_COMPLETE_INIT = 0.7

# SMP-REC 상관: 월별 로그수익률 ρ = -0.03 (통계적 비유의, 독립 처리)

# 실데이터 기반 GBM 파라미터 (SMP_REC_발전량_DATA_ANALYSIS.md 참조)
SMP_PARAMS = dict(
    initial_price=113.0,      # 원/kWh (2025 EPSIS 평균)
    annual_drift=0.02,        # +2%/년 (장기 +6.8%이나 최근 하락추세 반영, 보수적)
    annual_vol=0.32,          # 32.4% (월별 로그수익률 × √12)
    initial_volume=1000.0,    # MWh (미사용, 향후 확장용)
)
REC_PARAMS = dict(
    initial_price=77.0,       # 원/kWh (77,000원/REC ÷ 1000 kWh/MWh)
    annual_drift=0.02,        # +1.8%/년 → 반올림 +2% (결측치 보간 후 연도별 평균 변화율)
    annual_vol=0.28,          # 28.1% (결측치 선형보간 후 월별 로그수익률 × √12)
    initial_volume=500.0,     # MWh (미사용, 향후 확장용)
)

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
    monthly_vol=0.11,         # 탈계절화 월별 변동성 (일별 0.60 / √30 환산)
)


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

        # 롤오버 — 향후 확장
        self.pf_list = []
        self.rollover_schedule = []

        # 에피소드 관리 변수
        self.t = 0
        self.obj = 0.0
        self._init_market_buffers()
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
        self.state = self.build_state(self.init_state)
        self.history = []
        self.pf_list = []

    # ------------------------------------------------------------------ #
    #  시장 데이터 버퍼 (Task #4)                                         #
    # ------------------------------------------------------------------ #

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

    def _generate_market_step(self):
        """합성 시장 데이터를 한 달 전진시킴 (상관 GBM + 실데이터 계절성)."""
        dt = 1 / 12

        # 독립 브라운 운동 (SMP-REC 상관 ρ≈0, 독립 처리)
        z_smp = self.prng.randn()
        z_rec = self.prng.randn()

        # SMP 가격 — GBM
        last_smp = self.SMP_buffer[-1, 0]
        mu_s, sig_s = SMP_PARAMS['annual_drift'], SMP_PARAMS['annual_vol']
        smp_price = last_smp * np.exp(
            (mu_s - 0.5 * sig_s**2) * dt
            + sig_s * np.sqrt(dt) * z_smp
        )
        smp_volume = max(0.0, SMP_PARAMS['initial_volume']
                         * (1 + 0.1 * self.prng.randn()))

        # REC 가격 — GBM (SMP와 독립)
        last_rec = self.REC_buffer[-1, 0]
        mu_r, sig_r = REC_PARAMS['annual_drift'], REC_PARAMS['annual_vol']
        rec_price = last_rec * np.exp(
            (mu_r - 0.5 * sig_r**2) * dt
            + sig_r * np.sqrt(dt) * z_rec
        )
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
        """일별 베르누이 외생 이벤트 샘플링."""
        status = self.state.status_t
        u = self.prng.random()          # 일관성을 위해 항상 하나의 난수를 소비
        default_today = False
        curtail_today = False

        if status == 'PRE':
            default_today = (u < P_DEFAULT)
        elif status == 'POST':
            curtail_today = (u < P_CURTAIL)

        return {'default_today': default_today, 'curtail_today': curtail_today}

    # ------------------------------------------------------------------ #
    #  완공 확률 (Task #3)                                                #
    # ------------------------------------------------------------------ #

    def completion_probability_fn(self, P_prev, lambda_t=LAMBDA_DEFAULT):
        """위험률 기반 갱신: P_t = 1 - (1 - P_{t-1}) * exp(-lambda_t).

        추후 lambda_t를 보정된 함수로 교체 예정.
        """
        return 1.0 - (1.0 - P_prev) * np.exp(-lambda_t) # P = 생존확률이라고 보면 됨, LAMDA를 이제 상수로 뒀기 때문에 현실성은 당연히 없음. 

    # ------------------------------------------------------------------ #
    #  PV 엔진 — 단순 DCF 플레이스홀더 (Task #5)                         #
    # ------------------------------------------------------------------ #

    def pv_engine(self, status, time_t, curtail_today=False):
        """미래 운영 현금흐름의 현재가치.

        플레이스홀더: 시장 윈도우로부터 추정한 일정 월별 CF를
        잔여 운영기간에 대해 할인.
        내부 예측 로직은 모듈화되어 교체 가능.
        """
        if status == 'FAILED':
            return 0.0

        # 최근 시장 윈도우로부터 월별 CF 추정
        # SMP(원/kWh), REC(원/kWh), 발전량(MWh/월) → ×1000으로 kWh 변환
        avg_smp = np.mean(self.SMP_buffer[:, 0])
        avg_rec = np.mean(self.REC_buffer[:, 0])
        avg_energy = np.mean(self.energy_buffer)
        monthly_cf = (avg_smp + avg_rec) * avg_energy * 1000

        if status == 'PRE':
            months_to_start = max(0, T_CONSTRUCTION - time_t)
            remaining_op = T_OPERATION
        else:   # POST
            months_to_start = 0
            remaining_op = max(0, T_SINGLE_PF - time_t)

        if remaining_op <= 0:
            return 0.0

        # PV = 월별CF × 연금계수(잔여운영월) × 할인계수(운영시작까지)
        annuity = (1 - (1 + R_MONTHLY) ** (-remaining_op)) / R_MONTHLY
        discount = (1 + R_MONTHLY) ** (-months_to_start)
        pv = monthly_cf * annuity * discount

        # 출력제한: 해당월 운영 현금흐름 = 0
        if curtail_today and status == 'POST':
            pv -= monthly_cf

        return pv

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
    #  전이 함수 (Task #7)                                                #
    # ------------------------------------------------------------------ #

    def transition_fn(self, decision, exog_info):
        """S_t x A_t x W_t → S_{t+1}"""
        status = self.state.status_t
        P_prev = self.state.P_complete_t
        time_t = self.t

        # 1. 상태 갱신
        if exog_info['default_today'] and status == 'PRE':
            status = 'FAILED'
        elif status == 'PRE' and time_t >= T_CONSTRUCTION:
            status = 'POST'

        # 2. 시장 데이터 전진 (운영기간만 — 건설기간 중 시장 동결)
        if status == 'POST':
            self._generate_market_step()

        # 3. 완공 확률
        if status == 'FAILED':
            P_t = 0.0
        elif status == 'POST':
            P_t = 1.0
        else:
            P_t = self.completion_probability_fn(P_prev)

        # 4. PV
        PV_t = self.pv_engine(status, time_t, exog_info['curtail_today'])

        # 5. NAV
        if self.risk_adjust == 'NO':
            NAV_t = self.nav_calculator_rn(P_t, PV_t)
        else:
            NAV_t = self.nav_calculator_rw(P_t, PV_t)

        # 6. 롤오버 훅 (향후 확장)
        self.apply_events(time_t)

        return {
            'time_t': time_t,
            'status_t': status,
            'P_complete_t': P_t,
            'PV_t': PV_t,
            'NAV_t': NAV_t,
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
    #  롤오버 훅 (Task #10 — 현재 비어 있음)                              #
    # ------------------------------------------------------------------ #

    def apply_events(self, t):
        """PF 롤오버 로직을 위한 훅 함수.

        향후 용도:
          - t == K 시점에 신규 PF 편입
          - 토큰 발행 / 자본구조 변경
          - 수수료 차감
        """
        pass
