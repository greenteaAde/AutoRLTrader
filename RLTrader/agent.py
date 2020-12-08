import numpy as np
import utils


class Agent:
    """
    Attributes
    - initial_balance: 초기 투자금
    - balance: 현금 잔고
    - num_stocks: 보유 주식 수
    - portfolio_value: 포트폴리오 가치(투자금 잔고 + 주식 현재가 x 보유 주식 수)
    - base_portfolio_value: 직전 상태의 포트폴리오 가치, 현재 포트폴리오 가치와 비교하기 위함
    - num_buy: 매수 횟수
    - num_sell: 매도 횟수
    - num_hold: 관망 횟수
    - immediate_reward: 즉시 보상
    - profit_loss: 현재 손익
    - base_profit_loss: 직전 지연 보상 이후 손익
    - exploration_base: 탐험 행동 결정 기준 확률. 매수를 기조로 할지, 매도를 기조로 할지 정함

    Functions:
    - reset(): 에이전트의 상태를 초기화
    - set_balance(): 초기 자본금을 설정
    - get_status(): 에이전트 상태를 획득
    - decide_action(): 탐험 또는 정책 신경망에 의한 행동 결정
    - validate_action(): 헹동의 유효성 판단
    - decide_trading_unit(): 매수 또는 매도할 주식 수 결정
    - act(): 행동 수행
    """

    # 에이전트 상태의 차원
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 (일반적으로 0.015%)
    TRADING_TAX = 0.0025  # 거래세 (0.25%)

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망

    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self, environment, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold  # 지연보상 임계치

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        self.portfolio_value = 0  # 포트폴리오 가치
        self.base_portfolio_value = 0  # 직전 학습 시점의 포트폴리오 가치
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상
        self.profit_loss = 0  # 현재 손익
        self.base_profit_loss = 0  # 직전 지연 보상 이후 손익
        self.exploration_base = 0  # 탐험 행동 결정 기준

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset_exploration(self):
        self.exploration_base = 0.5 + np.random.rand() / 2

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        # 주식 보유 비율: 현재 상태에서 가장 많이 가질 수 있는 주식 수 대비 현재 보유한 주식의 비율
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        # 포트폴리오 가치 비율: 기준 포트폴리오 가치 대비 현재 포트폴리오 가치의 비율
        self.ratio_portfolio_value = (
                self.portfolio_value / self.base_portfolio_value)
        return (self.ratio_hold, self.ratio_portfolio_value)

    def decide_action(self, pred_value, pred_policy, epsilon):
        # 입력으로 들어온 epsilon의 확률로 무작위로 행동을 결정하고 그렇지 않은 경우 신경망을 통해 행동을 결정
        confidence = 0.  # ??

        pred = pred_policy
        if pred is None:
            pred = pred_value

        # 예측 값이 없을 경우 탐험
        if pred is None:
            epsilon = 1
        # 값이 모두 같은 경우 탐험
        else:
            max_pred = np.max(pred)
            if (pred == max_pred).all():
                epsilon = 1

        # 탐험 결정(0:매수, 1:매도)
        if np.random.rand() < epsilon:
            exploration = True
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        # 신용매수나 공매도는 고려하지 않기 때문에 이를 검증

        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (
                    1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        # 정책 신경망이 결정한 행동의 신뢰가 높을수록 매수 또는 매도하는 단위를 크게 저해줌
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit - self.min_trading_unit), 0)
        return self.min_trading_unit + added_trading

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        current_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                    self.balance - current_price * (1 + self.TRADING_CHARGE)
                    * trading_unit)

            # 보유 현금이 모자랄 경우 보유 현금 내에서 최대한 매수
            if balance < 0:
                trading_unit = max(min(
                    int(self.balance / (current_price * (1 + self.TRADING_CHARGE))),
                    self.max_trading_unit), self.min_trading_unit)

            # 수수료를 적용해 총 매수 금액 산정
            invest_amount = current_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)

            # 보유 주식이 모자랄 경우 보유 주식 내에서 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)

            # 수수료를 적용해 총 매도 금액 산정
            invest_amount = current_price * (1 - (self.TRADING_CHARGE + self.TRADING_TAX)) * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount
                self.num_stocks += trading_unit
                self.num_buy += 1

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1

        # 포트폴리오 가치 산정
        self.portfolio_value = self.balance + current_price * self.num_stocks
        self.profit_loss = (self.portfolio_value - self.initial_balance) / self.initial_balance

        # 즉시보상 : 수익률
        self.immediate_reward = self.profit_loss

        # 지연 보상: 익절, 손절 기준
        delayed_reward = 0
        self.base_profit_loss = (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value
        if self.base_profit_loss > self.delayed_reward_threshold or self.base_profit_loss < self.delayed_reward_threshold:
            # 목표 수익률을 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward