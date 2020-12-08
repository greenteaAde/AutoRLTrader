import os
import logging  # 학습 과정 중에서 정보를 기록하기 위함
import abc  # 추상 클래스를 정의하기 위함
import collections
import threading
import time
import numpy as np
from utils import sigmoid
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer


class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None, chart_data=None, training_data=None, min_trading_unit=1,
                 max_trading_unit=2, delayed_reward_threshold=.05, net='dnn', num_steps=1, lr=0.001,
                 value_network=None, policy_network=None, output_path='', reuse_models=True):
        """
        :param rl_method: 강화학습 기법, 'dqn','pg','ac','a2c','a3c'
        :param stock_code: 학습을 진행하는 주식 종목 코드
        :param chart_data: 주식 일봉 차트 데이터
        :param training_data: 전처리된 학습 데이터
        :param min_trading_unit: 투자 최소 단위
        :param max_trading_unit: 투자 최대 단위
        :param delayed_reward_threshold: 지연 보상 임곗값
        :param net: 신경망 종류, 'dnn','lstm','cnn'
        :param num_steps: LSTM, CNN 신경망에서 사용하는 샘플 묶음의 크기
        :param lr: learning rate
        :param value_network: 가치 신경망
        :param policy_network: 정책 신경망
        :param output_path: 가치 신경망과 정책 신경망 학습 과정 중 발생하는 로그,
                            가시화 결과 및 학습 종료 후 저장되는 신경망 모델 저장 경로
        :param reuse_models: 기존 모델 재활용 여부
        """
        # 인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 기법 설정
        self.rl_method = rl_method
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        # 에이전트 설정
        self.agent = Agent(self.environment,
                           min_trading_unit=min_trading_unit,
                           max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크키 = 학습 데이터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보
        self.loss = 0.
        self.exploration_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0
        # 로그 등 출력 경로
        self.output_path = output_path

    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        # 손익률을 회귀분석하는 모델
        if self.net == 'dnn':
            self.value_network = DNN(input_dim=self.num_features,
                                     output_dim=self.agent.NUM_ACTIONS,
                                     lr=self.lr, shared_network=shared_network,
                                     activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(input_dim=self.num_features,
                                             output_dim=self.agent.NUM_ACTIONS,
                                             lr=self.lr, num_steps=self.num_steps,
                                             shared_network=shared_network,
                                             activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(input_dim=self.num_features,
                                     output_dim=self.agent.NUM_ACTIONS,
                                     lr=self.lr, num_steps=self.num_steps,
                                     shared_network=shared_network,
                                     activation=activation, loss=loss)
        if self.reuse_models and os.path.exists:
            self.value_network.load_model(model_path=self.value_network_path)

    def init_policy_network(self, shared_network=None, activation='sigmoid', loss='mse'):
        # 샘플에 대하여 PV를 높이기 위해 취하기 좋은 행동에 대한 분류 모델
        if self.net == 'dnn':
            self.value_network = DNN(input_dim=self.num_features,
                                     output_dim=self.agent.NUM_ACTIONS,
                                     lr=self.lr, shared_network=shared_network,
                                     activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(input_dim=self.num_features,
                                             output_dim=self.agent.NUM_ACTIONS,
                                             lr=self.lr, num_steps=self.num_steps,
                                             shared_network=shared_network,
                                             activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(input_dim=self.num_features,
                                     output_dim=self.agent.NUM_ACTIONS,
                                     lr=self.lr, num_steps=self.num_steps,
                                     shared_network=shared_network,
                                     activation=activation, loss=loss)
        if self.reuse_models and os.path.exists:
            self.value_network.load_model(model_path=self.policy_network_path)

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.exploration_cnt = 0
        self.itr_cnt = 0
        self.batch_size = 0
        self.learning_cnt = 0

    def build_sample(self):
        # 차트 데이터의 현재 데이터에서 다음 데이터를 읽게 함
        # 학습 데이터의 다음 인덱스가 존재하는지 확인
        self.environment.observe()
        # sample을 "26개 값 + agent의 상태"의 28개의 값으로 구성
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        # 추상 메서드로서 ReinforcementLearner 클래스의 하위 클래스들은 반드시 이 함수를 구현해야 한다.
        # Reinforcement Learner을 상속하고도 이 추상 메서드를 구현하지 않으면 Notimplemented Exception 발생
        pass

    def update_networks(self, batch_size, delayed_reward, discount_factor):
        # get_batch 함수를 호출하여 배치 학습 데이터를 생성하고
        # 가치 신경망과 정책 신경망을 학습하기 위해 신경망 클래스의 train_on_batch 함수를 호출

        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신:
                loss += self.policy_network.train_on_batch(x, y_value)
            return loss
        return None

    def fit(self, delayed_reward, discount_factor, full=False):
        batch_size = len(self.memory_reward) if full else self.batch_size
        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            _loss = self.update_networks(batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1
                self.memory_learning_idx.append(self.training_data_idx)
            self.batch_size = 0

    def visualize(self, epoch_str, num_epoches, epsilon):
        # LSTM 신경망과 CNN 신경망을 사용하는 경우
        # 에이전트 행동, 보유 주식 수, 가치 신경망 출력, 정책 신경망 출력, 포트폴리오 가치가 환경의 일봉 수보다 (num_steps - 1)만큼 부족
        # (num_steps - 1)만큼 첫 부분에 채워줌
        self.memory_action += [Agent.ACTION_HOLD] * (self.num_steps - 1)
        self.memory_num_stocks += [0] * (self.num_steps - 1)
        if self.value_network is not None:
            self.memory_value += [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1)
        if self.policy_network is not None:
            self.memory_policy += [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1)
        self.memory_pv += [self.agent.initial_balance] * (self.num_steps - 1)

        self.visualizer.plot(epoch_str=epoch_str, num_epochs=num_epoches,
                             epsilon=epsilon, action_list=Agent.ACTIONS,
                             actions=self.memory_action,
                             num_stocks=self.memory_num_stocks,
                             outvals_value=self.memory_value,
                             outvals_policy=self.memory_policy,
                             exps=self.memory_exp_idx,
                             learning_idxes=self.memory_learning_idx,
                             initial_balance=self.agent.initial_balance,
                             pvs=self.memory_pv)
        self.visualizer.save(os.path.join(self.epoch_summary_dir, f'epoch_summary_{epoch_str}.png'))

    def run(self, num_epoches=100, balance=10000000, discount_factor=0.9, start_epsilon=0.5, learning=True):
        info = f'[{self.stock_code}] RL:{self.rl_method}'\
               f'Net:{self.net} DF:{discount_factor}'\
               f'TU:[{self.agent.min_trading_unit,self.agent.max_trading_unit}]'\
               f'DRT:{self.agent.delayed_reward_threshold}'
        with self.lock:
            logging.info(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않기 때문에 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0  # 에포크 중 가장 높은 포트폴리오 가치가 저장
        epoch_win_cnt = 0  # 수익이 발생한 에포크 수 저장

        # 학습 반복
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            # deque: 양방향 Queue
            q_sample = collections.deque(maxlen=self.num_steps)

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon * (1 - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon

            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break

                # num_steps 만큼 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))

                # 신경망 또는 탐험에 의한 행동 결정
                action, confidence, exploration = self.agent.decide_action(pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상솨 지연 보상 획득
                immediate_reward, delayed_reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0

                # 지연 보상이 발생한 경우 미니 배치 학습
                if learning and (delayed_reward != 0):
                    self.fit(delayed_reward, discount_factor)

            # 에포크 종료 후 학습
            if learning:
                self.fit(self.agent.profit_loss, discount_factor, full=True)

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info(f'[{self.stock_code}][Epoch {epoch_str}/{num_epoches}] Epsilon:{epsilon:.4f}'
                         f' #Expl.:{self.exploration_cnt}/{self.itr_cnt}'
                         f' #Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell}'
                         f' #Hold:{self.agent.num_hold} #Stocks:{self.agent.num_stocks}'
                         f' PV:{self.agent.portfolio_value:,.0f} LC:{self.learning_cnt}'
                         f' Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')

            # 에포크 관련 정보 가시화
            self.visualize(epoch_str, num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기혹
        with self.lock:
            logging.info(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f}'
                         f' Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')

    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)


class DQNLearner(ReinforcementLearner):
    # DQN: 가치 신경망으로만 강화학습을 하는 방식
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(reversed(self.memory_sample[-batch_size:]),
                     reversed(self.memory_action[-batch_size:]),
                     reversed(self.memory_value[-batch_size:]),
                     reversed(self.memory_reward[-batch_size:]))
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeroes((batch_size, self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, None


class PolicyGradientLearner(ReinforcementLearner):
    # 정책 경사 강화학습: 정책 신경망으로만 강화학습을 하는 방식
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(reversed(self.memory_sample[-batch_size:]),
                     reversed(self.memory_action[-batch_size:]),
                     reversed(self.memory_policy[-batch_size:]),
                     reversed(self.memory_reward[-batch_size:]))
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        reward_next = self.memory_reward[-1]
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_policy[i, action] = sigmoid(r)
            reward_next = reward
        return x, None, y_policy


class ActorCriticLearner(ReinforcementLearner):
    # 가치 신경망과 정책 신경망을 모두 사용하는 강화학습 방법
    def __init__(self, *args, shared_network=None, value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(net=self.net, num_steps=self.num_steps,
                                                             input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(reversed(self.memory_sample[-batch_size:]),
                     reversed(self.memory_action[-batch_size:]),
                     reversed(self.memory_value[-batch_size:]),
                     reversed(self.memory_policy[-batch_size:]),
                     reversed(self.memory_reward[-batch_size:]))
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            y_policy[i, action] = sigmoid(r)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy


class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(reversed(self.memory_sample[-batch_size:]),
                     reversed(self.memory_action[-batch_size:]),
                     reversed(self.memory_value[-batch_size:]),
                     reversed(self.memory_policy[-batch_size:]),
                     reversed(self.memory_reward[-batch_size:]))
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            advantage = value[action] - value.mean()
            y_policy[i, action] = sigmoid(advantage)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy


class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None, list_chart_data=None,
                 list_training_data=None, list_min_trading_unit=None, list_max_trading_unit=None,
                 value_network_path=None, policy_network_path=None, **kwargs):
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(net=self.net, num_steps=self.num_steps,
                                                         input_dim=self.num_features)
        self.value_network_path=value_network_path
        self.policy_network_path=policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data, min_trading_unit, max_trading_unit) in zip(
                list_stock_code, list_chart_data, list_training_data, list_min_trading_unit, list_max_trading_unit):
            learner = A2CLearner(*args, stock_code=stock_code, chart_data=chart_data,
                                 training_data=training_data,
                                 min_trading_unit=min_trading_unit,
                                 max_trading_unit=max_trading_unit,
                                 shared_network=self.shared_network,
                                 value_network=self.value_network,
                                 policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)

    def run(self, num_epoches=100, balance=10000000, discount_factor=0.9, start_epsilon=0.9, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(target=learner.fit, daemon=True,
                                            kwargs={'num_epoches': num_epoches, 'balance': balance,
                                                    'discount_factor': discount_factor,
                                                    'start_epsilon': start_epsilon, 'learning': learning}))
        # A2C를 병렬로 동시에 수행
        # 가치 신경망과 정책 신경망을 공유하면서 이들을 동시에 학습시킴
        # A2CLearner 클래스 객체는 하나의 주식 종목 환경애서 탐험히며 손익률을 높이는 방향으로 가치/정책 신경망 학습 수행
        # 스레드를 이용하여 각 A2CLearner 객체의 run() 함수를 수행
        # 모든 A2C 강화학습을 마칠 때까지 기다린 후 최종적으로 A3C 강화학습을 마침
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads:
            thread.join()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(reversed(self.memory_sample[-batch_size:]),
                     reversed(self.memory_action[-batch_size:]),
                     reversed(self.memory_value[-batch_size:]),
                     reversed(self.memory_policy[-batch_size:]),
                     reversed(self.memory_reward[-batch_size:]))
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            advantage = value[action] - value.mean()
            y_policy[i, action] = sigmoid(advantage)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy