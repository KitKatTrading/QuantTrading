import os
import pandas as pd
import importlib

# Adjust dir_data and add dir_utils to the Python path
current_script_dir = os.path.dirname(__file__)
dir_data = os.path.join(current_script_dir, '..', 'module_data')
dir_data = os.path.normpath(dir_data)
dir_utils = os.path.join(current_script_dir)


class Strategy:
    def __init__(self, name_symbol, data_source, name_strategy, timeframe_high, timeframe_mid, timeframe_low,
                 function_high_timeframe, function_mid_timeframe, function_low_timeframe):
        self.name_symbol = name_symbol
        self.data_source = data_source
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

        # Dynamically import the high timeframe strategy module
        self.high_strategy_module = importlib.import_module(f"timeframe_high.{self.high_strategy_file}")

    def high_timeframe_analysis(self):
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_high + '.csv')
        pd_OHLC_high = pd.read_csv(file_path, index_col=0)

        # Call the 'main' function from the strategy module
        return self.function_high_timeframe.main(pd_OHLC_high)

    def mid_timeframe_analysis(self):
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
        pd_OHLC_mid = pd.read_csv(file_path, index_col=0)

        # Call the 'main' function from the strategy module
        return self.function_mid_timeframe.main(pd_OHLC_mid)

if __name__ == '__main__':
    # Example usage
    strategy_chanlun_12h = Strategy(name_symbol='BTCUSDT', data_source='binance', name_strategy='chanlun_12h',
                                    timeframe_high='1w', timeframe_mid='12h', timeframe_low='1h',
                                    high_strategy_file='SMA_5_10_20_trend')

    high_timeframe_decision = strategy_chanlun_12h.high_timeframe_analysis()
    print(high_timeframe_decision)