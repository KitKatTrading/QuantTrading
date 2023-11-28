import os
import pandas as pd
import importlib

from strategy import Strategy

# Adjust dir_data and add dir_utils to the Python path
current_script_dir = os.path.dirname(__file__)
dir_data = os.path.join(current_script_dir, '', 'module_data')
dir_data = os.path.normpath(dir_data)

class Backtesting:

    def __init__(self, name_symbol, data_source, name_strategy, timeframe_high, timeframe_mid, timeframe_low,
                 function_high_timeframe, function_mid_timeframe, function_low_timeframe,
                 bt_start_date, bt_end_date):
        self.name_symbol = name_symbol
        self.data_source = data_source
        self.bt_start_date = bt_start_date
        self.bt_end_date = bt_end_date
        self.name_strategy = name_strategy
        self.timeframe_high = timeframe_high
        self.timeframe_mid = timeframe_mid
        self.timeframe_low = timeframe_low
        self.function_high_timeframe = function_high_timeframe
        self.function_mid_timeframe = function_mid_timeframe
        self.function_low_timeframe = function_low_timeframe

        # Set data directory based on data source
        if self.data_source == 'binance':
            self.data_dir = os.path.join(dir_data, 'data_binance')

        # Define the strategy object
        self.strategy = Strategy(name_symbol=self.name_symbol, data_source=self.data_source, name_strategy=self.name_strategy,
                                 timeframe_high=self.timeframe_high, timeframe_mid=self.timeframe_mid, timeframe_low=self.timeframe_low,
                                 function_high_timeframe=self.function_high_timeframe,
                                 function_mid_timeframe=self.function_mid_timeframe,
                                 function_low_timeframe=self.function_low_timeframe)

    def run_backtesting(self):

        # Load the OHLC data for the high timeframe
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_high + '.csv')
        pd_OHLC_high = pd.read_csv(file_path, index_col=0)
        pd_OHLC_high = pd_OHLC_high[(pd_OHLC_high['date'] >= self.bt_start_date) & (pd_OHLC_high['date'] <= self.bt_end_date)]

        # Load the OHLC data for the mid timeframe
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
        pd_OHLC_mid = pd.read_csv(file_path, index_col=0)
        pd_OHLC_mid = pd_OHLC_mid[(pd_OHLC_mid['date'] >= self.bt_start_date) & (pd_OHLC_mid['date'] <= self.bt_end_date)]

        # Load the OHLC data for the low timeframe
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_low + '.csv')
        pd_OHLC_low = pd.read_csv(file_path, index_col=0)
        pd_OHLC_low = pd_OHLC_low[(pd_OHLC_low['date'] >= self.bt_start_date) & (pd_OHLC_low['date'] <= self.bt_end_date)]

        # Run the direction module
        self.strategy.high_timeframe_analysis.main(pd_OHLC_high, run_mode='backtest')

        # Run the pattern module
        self.strategy.mid_timeframe_analysis.main(pd_OHLC_mid, run_mode='backtest')

        # Run the entry module
        self.strategy.low_timeframe_analysis.main(pd_OHLC_low, run_mode='backtest')



if __name__ == '__main__':

    # Define the backtesting object
    backtesting = Backtesting(name_symbol='BTCUSDT', data_source='binance', name_strategy='chanlun_12h',
                              timeframe_high='1w', timeframe_mid='12h', timeframe_low='1h',
                              function_high_timeframe='SMA_5_10_20_trend',
                              function_mid_timeframe='chanlun_central_hub',
                              function_low_timeframe='RSI_divergence',
                              bt_start_date='2021-01-01', bt_end_date='2021-01-31')

    # Run the backtesting
    trading_decision = backtesting.run_backtesting()

    print(trading_decision)

    print('Done!')