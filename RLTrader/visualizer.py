import numpy as np
import threading
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from agent import Agent
plt.switch_backend('agg')

lock = threading.Lock()
# A3C 강화학습 시 여러 스레드가 병렬로 강화학습을 수행하기 때문에
# 안정적인 가시화를 위해 가시화 작업 도중 다른 스레드의 간섭을 받지 않게 한 것


class Visualizer:
    COLORS = ['r', 'b', 'g']

    def __init__(self, vnet=False):
        self.canvas = None
        self.fig = None
        self.axes = None
        self.title = ''

    def prepare(self, chart_data, title):
        self.title = title
        with lock:
            self.fig, self.axes = plt.subplots(nrows=5, ncols=1, facecolor='w', sharex=True)
            for ax in self.axes:
                # 보기 어려운 과학적 표기 비활성화
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                ax.yaxis.tick_right()

            # 1번째 그래프: 종목일봉차트
            self.axes[0].set_ylabel('Env.')
            x = np.arange(len(chart_data))
            ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))
            candlestick_ohlc(self.axes[0], ohlc, colorup='r', colordown='b')
            ax = self.axes[0].twinx()
            volume = np.array(chart_data)[:, -1].tolist()
            ax.bar(x, volume, color='b', alpha=0.3)

    def plot(self, epoch_str=None, num_epochs=None, epsilon=None, action_list=None,
             actions=None, num_stocks=None, outvals_value=[], outvals_policy=[],
             exps=None, learning_idxes=None, initial_balance=None, pvs=None):
        with lock:
            x = np.arange(len(actions))
            actions = np.array(actions)
            outvals_value = np.array(outvals_value)  # 가치 신경망의 출력
            outvals_policy = np.array(outvals_policy)  # 정책 신경망의 출력
            pvs_base = np.zeros(len(actions)) + initial_balance  # 초기 자본금

            # 2번째 그래프: 보유 주식 수 및 매수(빨강), 매도(파랑) 여부
            for action, color in zip(action_list, self.COLORS):
                for i in x[actions==action]:
                    self.axes[1].axvline(i, color=color, alpha=0.1)
            self.axes[1].plot(x, num_stocks, '-k')

            # 3번쨰 그래프: 가치 신경망
            if len(outvals_value) > 0:
                max_actions = np.argmax(outvals_value, axis=1)
                for action, color in zip(action_list, self.COLORS):
                    # 배경 그리기
                    for idx in x:
                        # 매수: 빨강, 매도: 파랑, 관망: 초록
                        # 가장 예측치가 높은 행동에 대한 색으로 칠함
                        if max_actions[idx] == action:
                            self.axes[2].axvline(idx, color=color, alpha=0.1)
                    # 가치 신경망 출력의 tanh 그리기
                    # 행동에 대한 예측 가치를 라인 차트로 그린다.
                    self.axes[2].plot(x, outvals_value[:, action], color=color, linestyle='-')

            # 4번째 그래프: 정책 신경망
            # 탐혐을 노란색 배경으로 그리기
            for exp_idx in exps:
                self.axes[3].axvline(exp_idx, color='y')
            # 행동을 배경으로 그리기
            _outvals = outvals_policy if len(outvals_policy) > 0 else outvals_value
            for idx, outval in zip(x, _outvals):
                color = 'white'
                if np.isnan(outval.max()):
                    continue
                if outval.argmax() == Agent.ACTION_BUY:
                    color = 'r'  # 매수 빨간색
                elif outval.argmax() == Agent.ACTION_SELL:
                    color = 'b'  # 매도 파란색
                self.axes[3].axvline(idx, color=color, alpha=0.1)
            # 정책신경망의 출력 그리기
            # 빨간 선이 파란선보다 위인 경우 매수, 반대는 매도
            if len(outvals_policy) > 0:
                for action, color in zip(action_list, self.COLORS):
                    self.axes[3].plot(x, outvals_policy[:, action], color=color, linestyle='-')

            # 5번째 그래프: 포트폴리오 가치
            # 초기 자본금을 가로로 일직선을 그어서 손익을 쉽게 파악하도록 함
            self.axes[4].axhline(initial_balance, linestyle='-', color='gray')
            # 포트폴리오 가치가 초기 자본금보다 높으면 빨간색, 낮으면 파란색
            self.axes[4].fill_between(x, pvs, pvs_base, where=pvs > pvs_base, facecolor='r', alpha=0.1)
            self.axes[4].fill_between(x, pvs, pvs_base, where=pvs < pvs_base, facecolor='b', alpha=0.1)
            # 포트폴리오 가치를 실선으로 그리기
            self.axes[4].plot(x, pvs, '-k')

            # 학습 위치 표시를 노란색으로 표시
            for learning_idx in learning_idxes:
                self.axes[4].axvline(learning_idx, color='y')

            # 에포크 및 탐험 비율
            self.fig.suptitle(f'{self.title} \nEpoch:{epoch_str}/{num_epochs} e={epsilon:.2f}')
            # 캔버스 레이아웃 조정
            self.fig.tight_layout()  # Figure 크기에 알맞게 내부 차트의 크기를 조정해 줍니다.
            self.fig.subplots_adjust(top=0.85)

    def clear(self, xlim):
        with lock:
            _axes = self.axes.tolist()
            for ax in _axes[1:]:
                ax.cla()  # 그린 차트 지우기
                ax.relim()  # limit를 초기화
                ax.autoscale()  # 스케일 재설정
            # y축 레이블 재설정
            self.axes[1].set_ylabel('Agent')
            self.axes[2].set_ylabel('V')
            self.axes[3].set_ylabel('P')
            self.axes[4].set_ylabel('PV')
            for ax in _axes:
                ax.set_xlim(xlim)  # x축 limit 재설정
                ax.get_xaxis().get_major_formatter().set_scientific(False)  # 과학적 표기 비활성화
                # x축 간격을 일정하게 설정
                # 주말이나 공휴일처럼 휴장하는 날에는 차트가 비기 때문
                ax.ticklabel_format(useOffset=False)

    def save(self, path):
        with lock:
            self.fig.savefig(path)
