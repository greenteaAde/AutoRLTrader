import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager

# 다양항 조건으로 가화학습을 수행할 수 있게 프로그램 인자를 구성하여
# 입력받은 인자에 따라 학습기 클래스를 이용해 강화학습을 수행하고
# 학습한 신경망들을 저장하는 메인 모듈

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--ver', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c'], default='ac')
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn'], default='lstm')
    parser.add_argument('--num_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=0)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--delayed_reward_threshold', type=float, default=0.05)
    parser.add_argument('--backend', choices=['tensorflow', 'plaidml'], default='tensorflow')
    parser.add_argument('--output_name', default=utils.get_time_str())
    parser.add_argument('--value_network_name')
    parser.add_argument('--policy_network_name')
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_true')
    parser.add_argument('--start_date', default='20170101')
    parser.add_argument('--end_date', default='20171231')
    args = parser.parse_args()

    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 설정
    output_path = os.path.join(settings.BASE_DIR, f'output/{args.output_name}_{args.rl_method}_{args.net}')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    """
    json: Javascript Object Notation
    파이썬의 딕셔너리와 호환이 잘 됨
    load()함수와 dumps()함수로 json 문자열에서 dict로, dict에서 json 문자열로 변환 가능 
    """
    """
    ArgumentParser 객체에 저장돼 있는 프로그램 인자들은 내장함수 vars()로 딕션어리로 변환 가능
    """
    # 로그 기록 설정
    file_handler = logging.FileHandler(filename=os.path.join(output_path, f'{args.output_name}.log'), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', handlers=[file_handler, stream_handler], level=logging.DEBUG)

    from agent import Agent
    from learners import DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR, f'models/{args.value_network_name}.h5')
    else:
        value_network_path = os.path.join(output_path, f'{args.rl_method}_{args.net}_value_{args.output_name}.h5')
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR, f'models/{args.policy_network_name}.h5')
    else:
        policy_network_path = os.path.join(output_path, f'{args.rl_method}_{args.net}_policy_{args.output_name}.h5')

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            os.path.join(settings.BASE_DIR, f'data/{args.ver}/{stock_code}.csv'),
            args.start_date, args.end_date, ver = args.ver)

        # 최소/최대 투자 단위 설정
        min_trading_unit = max(int(100000 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(1000000 / chart_data.iloc[-1]['close']), 1)

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method,
                         'delayed_reward_threshold': args.delayed_reward_threshold,
                         'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
                         'output_path': output_path, 'reuse_models': args.reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                                  'chart_data': chart_data,
                                  'training_data': training_data,
                                  'min_trading_unit': min_trading_unit,
                                  'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params,
                                                'value_network_path': value_network_path,
                                                'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params,
                                        'value_network_path': value_network_path,
                                        'policy_network_path': policy_network_path})
            if learner is not None:
                learner.run(balance=args.balance,
                            num_epoches=args.num_epoches,
                            discount_factor=args.discount_factor,
                            start_epsilon=args.start_epsilon,
                            learning=args.learning)
                learner.save_models()
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)

