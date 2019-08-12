# 아래  .py 파일들은  해당 PC   Lib / site-packages  폴더에 저장하기. 

import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner
from pandas_datareader import data 
import fix_yahoo_finance
import pandas as pd

fix_yahoo_finance.pdr_override()

if __name__ == '__main__':
    stock_code = '068270'  # 학습시킬 종목 코드 입력하기 

    # 학습 기록 남기기 
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()
    if not os.path.exists('logs/%s' % stock_code):
        os.makedirs('logs/%s' % stock_code)
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
                        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    #  야후 사이트에서 주식데이터 불러오기 
    chart_data = data.get_data_yahoo('068270.KS','2015-11-02','2018-06-29')
    chart_data = chart_data.drop('Close', 1) # close 열 제거 
    chart_data = chart_data.reset_index()

    chart_data = chart_data.rename(columns = {'Date' : 'date', 'Open' : 'open', 'High' : 'high', "Low" : 'low',  "Volume" : 'volume',
                            'Adj Close' : 'close'})
    

    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터
    training_data = training_data[(training_data['date'] >= '2015-11-02') &
                                  (training_data['date'] <= '2017-12-29')]
    training_data = training_data.dropna()
    #외부변수 불러오기 
    external_data = pd.read_csv('non_financial.csv')
    # 차트 데이터와 학습 데이터 분리. 
    training_data = pd.concat([training_data, external_data], axis=0, ignore_index=True)
    features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
    chart_data = training_data[features_chart_data]
    # 학습 데이터 분리
    features_training_data = [
        'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
        'close_lastclose_ratio', 'volume_lastvolume_ratio',
        'close_ma5_ratio', 'volume_ma5_ratio',
        'close_ma10_ratio', 'volume_ma10_ratio',
        'close_ma20_ratio', 
        'close_ma60_ratio', 'volume_ma60_ratio',
        'number_of_RA_patinets', 'number_of_HC_patients',
        'number_of_articles'

    ]
    training_data = training_data[features_training_data]   # 여기까진 실행 된다. 

    # 강화학습 시작
    policy_learner = PolicyLearner(
        stock_code = stock_code, chart_data = chart_data, training_data = training_data,
        min_trading_unit=10, max_trading_unit=15, delayed_reward_threshold=0.05, lr=0.0001)
    policy_learner.fit(balance = 10000000, num_epoches = 1500,
                       discount_factor=0, start_epsilon=.5)

    # 학습한 정책 신경망 저장   -> 추후에 model 폴더에 들어가서 TEST할때, 해당 신경망 파일을 불러오기. 
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % stock_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_%s.h5' % timestr)
    policy_learner.policy_network.save_model(model_path)
