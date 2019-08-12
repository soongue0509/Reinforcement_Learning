import logging
import os
import settings
import data_manager
from policy_learner import PolicyLearner
from pandas_datareader import data as pdr

if __name__ == '__main__':
    stock_code = '068270'  # 셀트리온
    model_ver = '20190227102252'


    # 로그 기록
    log_dir = os.path.join(settings.BASE_DIR, 'logs/%s' % stock_code)
    timestr = settings.get_time_str()
    file_handler = logging.FileHandler(filename=os.path.join(
        log_dir, "%s_%s.log" % (stock_code, timestr)), encoding='utf-8')
    stream_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s",
        handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 주식 데이터 준비
    chart_data = pdr.get_data_yahoo('068270.KS','2015-11-02','2018-06-29')
    chart_data = chart_data.drop('Close', 1) # close 열 제거 
    chart_data = chart_data.reset_index()

    chart_data = chart_data.rename(columns = {'Date' : 'date', 'Open' : 'open', 'High' : 'high', "Low" : 'low',  "Volume" : 'volume',
                            'Adj Close' : 'close'})
    prep_data = data_manager.preprocess(chart_data)
    training_data = data_manager.build_training_data(prep_data)

    # 기간 필터링
    training_data = training_data[(training_data['date'] >= '2017-07-01') &
                                  (training_data['date'] <= '2018-06-29')]
    training_data = training_data.dropna()

    # 차트 데이터 분리
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
    training_data = training_data[features_training_data]
    
    
    
    # 비 학습 투자 시뮬레이션 시작
    policy_learner = PolicyLearner(
        stock_code=stock_code, chart_data=chart_data, training_data=training_data,
        min_trading_unit=1, max_trading_unit=3)
    policy_learner.trade(balance=10000000,
                         model_path=os.path.join(
                             settings.BASE_DIR,
                             'models/{}/model_{}.h5'.format(stock_code, model_ver)))
    
    
    
    
    