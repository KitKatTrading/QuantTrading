import os
import pandas as pd
import importlib

# read the config file "config_local_path.py"
from config import config_local_path

PATH_DATA = config_local_path.gvars['dir_module_data_crypto_binance_live']

class Strategy:
    def __init__(self, name_symbol, data_source, name_strategy, timeframe_high, timeframe_mid, timeframe_low,
                 function_high_timeframe, function_mid_timeframe, function_low_timeframe):

        self.chart_entry = None
        self.chart_pattern = None
        self.name_symbol = name_symbol
        self.data_source = data_source
        self.name_strategy = name_strategy
        self.timeframe_high = timeframe_high
        self.timeframe_mid = timeframe_mid
        self.timeframe_low = timeframe_low
        self.function_high_timeframe = function_high_timeframe
        self.function_mid_timeframe = function_mid_timeframe
        self.function_low_timeframe = function_low_timeframe
        self.high_timeframe_analysis = None
        self.mid_timeframe_analysis = None
        self.low_timeframe_analysis = None
        self.decision_direction = 0
        self.decision_pattern = 0
        self.decision_entry = 0

        # Set data directory based on data source
        if self.data_source == 'binance':
            self.data_dir = os.path.join(PATH_DATA, 'data_binance')

        # Dynamically import the decision functions for H/M/L timeframes
        self.strategy_high_timeframe = importlib.import_module(f"module_direction.{self.function_high_timeframe}")
        self.strategy_mid_timeframe = importlib.import_module(f"module_pattern.{self.function_mid_timeframe}")
        self.strategy_low_timeframe = importlib.import_module(f"module_entry.{self.function_low_timeframe}")

    def run_direction_module_live(self, use_default_data=True, df_OHLC_high=None):

        if use_default_data:
            file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_high + '.csv')
            df_OHLC_high = pd.read_csv(file_path, index_col=0)
        else:
            df_OHLC_high = df_OHLC_high

        # Call the 'main' function from the strategy module
        self.decision_direction = self.strategy_high_timeframe.main(df_OHLC_high)

    def run_pattern_module(self, use_default_data=True, df_OHLC_mid=None):

        if use_default_data:
            file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
            df_OHLC_mid = pd.read_csv(file_path, index_col=0)
        else:
            df_OHLC_mid = df_OHLC_mid

        # Call the 'main' function from the strategy module
        self.decision_pattern, self.chart_pattern = self.strategy_mid_timeframe.main(df_OHLC_mid,
                                                                                     name_symbol=self.name_symbol,
                                                                                     time_frame=self.timeframe_mid)

    def run_entry_module(self, use_default_data=True, df_OHLC_low=None):

        if use_default_data:
            file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_low + '.csv')
            df_OHLC_low = pd.read_csv(file_path, index_col=0)
        else:
            df_OHLC_low = df_OHLC_low

        # Call the 'main' function from the strategy module
        self.decision_entry, self.chart_entry = self.strategy_low_timeframe.main(df_OHLC_low)

    def check_ultimate_decision_all_modules(self):
        """ This function checks the trading decision based on all three modules"""
        trading_decision = 0


        # Use try / except to skip situation where not enough candels are available for analysis
        try:
            # directional module analysis
            self.run_direction_module_live()

            # DEBUG - run entry module before even needed to check if it works
            self.run_entry_module()

            # only run pattern analysis if directional module analysis is not 0
            if self.decision_direction != 0:
                self.run_pattern_module()

                # only run trade entry module if both directional and pattern modules are meaningful:
                if self.decision_pattern * self.decision_direction == 1:
                    self.run_entry_module()

                    # Checking long setup opportunity:
                    if self.decision_direction == 1 and self.decision_pattern == 1 and self.decision_entry == 1:
                        # print('Long trading opportunity')
                        trading_decision = 1

                    # Checking short setup opportunity:
                    elif self.decision_direction == -1 and self.decision_pattern == -1 and self.decision_entry == -1:
                        # print('Short trading opportunity')
                        trading_decision = -1

                    # No setup opportunity:
                    else:
                        # print('No trading opportunity')
                        trading_decision = 0

        except:
            pass

        return trading_decision

    def check_trading_setup(self):
        """ This function checks the trading setup opportunity, without considering the entry module"""
        trading_setup = 0

        # directional module analysis
        self.run_direction_module_live()

        # only run pattern analysis if directional module analysis is not 0
        if self.decision_direction != 0:
            self.run_pattern_module()

            # if both directional and pattern modules are ready, update watchlist
            if self.decision_pattern * self.decision_direction == 1:
                trading_setup = 1

        return trading_setup


if __name__ == '__main__':

    # Example usage
    strategy_chanlun_12h = Strategy(name_symbol='BTCUSDT', data_source='binance', name_strategy='chanlun_12h',
                                    timeframe_high='1w', timeframe_mid='12h', timeframe_low='1h',
                                    function_high_timeframe='SMA_5_10_20_trend',
                                    function_mid_timeframe='chanlun_central_hub',
                                    function_low_timeframe='RSI_divergence')

    strategy_chanlun_12h.check_ultimate_decision_all_modules()
    # print(strategy_chanlun_12h.decision_direction)
    # print(strategy_chanlun_12h.decision_pattern)
    # print(strategy_chanlun_12h.decision_entry)



