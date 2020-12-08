import pandas as pd
import numpy as np

COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

# 차트 데이터에서 전처리로 얻을 수 있는 자질
COLUMNS_TRANING_DATA_V1 = ['open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
                           'close_lastclose_ratio', 'volume_lastvolume_ratio',
                           'close_ma5_ratio', 'volume_ma5_ratio',
                           'close_ma10_ratio', 'volume_ma10_ratio',
                           'close_ma20_ratio', 'volume_ma20_ratio',
                           'close_ma60_ratio', 'volume_ma60_ratio',
                           'close_ma120_ratio', 'volume_ma120_ratio']

# 차트 데이터 외에도 기본적 분석 지표와 코스피지수, 국채 3년 데이터 추가
COLUMNS_TRANING_DATA_V2 = ['per', 'pbr', 'roe'
                           'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
                           'close_lastclose_ratio', 'volume_lastvolume_ratio',
                           'close_ma5_ratio', 'volume_ma5_ratio',
                           'close_ma10_ratio', 'volume_ma10_ratio',
                           'close_ma20_ratio', 'volume_ma20_ratio',
                           'close_ma60_ratio', 'volume_ma60_ratio',
                           'close_ma120_ratio', 'volume_ma120_ratio',
                           'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio',
                           'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio',
                           'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio',
                           'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio']

def preprocessing(data):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
        data[f'volume_ma{window}_ratio'] = (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']

    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = (data['open'][1:].values - data['close'][:-1].values) \
                                           / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = (data['close'][1:].values - data['close'][:-1].values) \
                                            / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (data['volume'][1:] - data['volume'][:-1].values) \
                                              / data['volume'][:-1] \
                                                  .replace(to_replace=0, method='ffill') \
                                                  .replace(to_replace=0, method='bfill')
    return data


def load_data(fpath, date_from, date_to, ver='v2'):
    header = None if ver == '1' else 0
    data = pd.read_csv(fpath, thousands=',', header=header, converters={'date': lambda x: str(x)})

    # 데이터 전처리
    data = preprocessing(data)

    # 기간 필터링
    data['date'] = data['date'].str.replace('-', '')
    data = data[(data['date'] >= date_from) & (data['date'] <= date_to)]
    data = data.dropna()

    # 차트 데이터 분리
    chart_data = data[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = data[COLUMNS_TRANING_DATA_V1]
    elif ver == 'v2':
        data.loc[:, ['per', 'pbr', 'roe']] = data[['per', 'pbr', 'roe']].apply(lambda x: x/100)
        training_data = data[COLUMNS_TRANING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception('Invalid Error')

    return chart_data, training_data
