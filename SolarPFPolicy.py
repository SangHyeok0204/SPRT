"""
SolarPFPolicy.py — 최소 no-op 정책

MDP BasePolicy 패턴 (MDP_Framework_Structure.md)
본 프로젝트에서는 실질적 의사결정 없음; 확장성을 위해 프레임워크 유지.
"""

from copy import copy


class SolarPFPolicy:
    """MDP BasePolicy 패턴을 따르는 최소 no-op 정책."""

    def __init__(self, model, policy_type='no_op'):
        self.M = model
        self.policy_type = policy_type

    # ------------------------------------------------------------------ #
    #  정책 디스패처                                                       #
    # ------------------------------------------------------------------ #

    def get_decision(self):
        """의사결정 반환. 모든 분기에서 build_decision(dict)을 반환."""
        if self.policy_type == 'no_op':
            return self.M.build_decision({'no_op': 0})
        # 향후: 롤오버 타이밍 정책 등 추가 가능
        return self.M.build_decision({'no_op': 0})

    # ------------------------------------------------------------------ #
    #  에피소드 실행기 (MDP 패턴 3)                                       #
    # ------------------------------------------------------------------ #

    def run_policy(self, seed=None):
        """에피소드 1회 실행 후 (최종_목적값, 이력)을 반환.

        매개변수
        ----------
        seed : int 또는 None — 지정 시 이번 실행의 모델 시드를 오버라이드.
        """
        model_copy = copy(self.M)
        model_copy.reset(seed=seed)
        policy_copy = SolarPFPolicy(model_copy, self.policy_type)

        for _ in range(model_copy.T):
            decision = policy_copy.get_decision()
            model_copy.step(decision)

        return (model_copy.obj, model_copy.history.copy(), model_copy.merge_history.copy())

    # ------------------------------------------------------------------ #
    #  헬퍼                                                               #
    # ------------------------------------------------------------------ #

    def _get_params(self):
        """정책 파라미터 추출 (모델 참조 제외)."""
        return {k: v for k, v in self.__dict__.items() if k != 'M'}
