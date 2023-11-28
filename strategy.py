import os
import pandas as pd
import importlib

# Adjust dir_data and add dir_utils to the Python path
current_script_dir = os.path.dirname(__file__)
dir_data = os.path.join(current_script_dir, '', 'module_data')
dir_data = os.path.normpath(dir_data)


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
        self.high_timeframe_analysis = None
        self.mid_timeframe_analysis = None
        self.low_timeframe_analysis = None
        self.direction_module_decision = 0
        self.pattern_module_decision = 0
        self.entry_module_decision = 0


        # Set data directory based on data source
        if self.data_source == 'binance':
            self.data_dir = os.path.join(dir_data, 'data_binance')

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
        self.direction_module_decision = self.strategy_high_timeframe.main(df_OHLC_high)

    def run_pattern_module(self, use_default_data=True, df_OHLC_mid=None):

        if use_default_data:
            file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
            df_OHLC_mid = pd.read_csv(file_path, index_col=0)
        else:
            df_OHLC_mid = df_OHLC_mid

        # Call the 'main' function from the strategy module
        self.pattern_module_decision = self.strategy_mid_timeframe.main(df_OHLC_mid)

    def run_entry_module(self, use_default_data=True, df_OHLC_low=None):

        if use_default_data:
            file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_low + '.csv')
            df_OHLC_low = pd.read_csv(file_path, index_col=0)
        else:
            df_OHLC_low = df_OHLC_low

        # Call the 'main' function from the strategy module
        self.entry_module_decision = self.strategy_low_timeframe.main(df_OHLC_low)

    def check_ultimate_decision_all_modules(self):
        """ This function checks the trading decision based on all three modules"""
        trading_decision = 0

        # directional module analysis
        self.run_direction_module_live()

        # only run pattern analysis if directional module analysis is not 0
        if self.direction_module_decision != 0:
            self.run_pattern_module()

            # only run trade entry module if both directional and pattern modules are meaningful:
            if self.pattern_module_decision * self.direction_module_decision == 1:
                self.run_entry_module()

                # Checking long setup opportunity:
                if self.direction_module_decision == 1 and self.pattern_module_decision == 1 and self.entry_module_decision == 1:
                    print('Long trading opportunity')
                    trading_decision = 1

                # Checking short setup opportunity:
                elif self.direction_module_decision == -1 and self.pattern_module_decision == -1 and self.entry_module_decision == -1:
                    print('Short trading opportunity')
                    trading_decision = -1

                # No setup opportunity:
                else:
                    print('No trading opportunity')

        return trading_decision

    def check_trading_setup(self):
        """ This function checks the trading setup opportunity, without considering the entry module"""
        trading_setup = 0

        # directional module analysis
        self.run_direction_module_live()

        # only run pattern analysis if directional module analysis is not 0
        if self.direction_module_decision != 0:
            self.run_pattern_module()

            # if both directional and pattern modules are ready, update watchlist
            if self.pattern_module_decision * self.direction_module_decision == 1:
                trading_setup = 1

        return trading_setup




if __name__ == '__main__':

    # Example usage
    strategy_chanlun_12h = Strategy(name_symbol='BTCUSDT', data_source='binance', name_strategy='chanlun_12h',
                                    timeframe_high='1w', timeframe_mid='12h', timeframe_low='1h',
                                    function_high_timeframe='SMA_5_10_20_trend',
                                    function_mid_timeframe='chanlun_central_hub',
                                    function_low_timeframe='RSI_divergence')

    strategy_chanlun_12h.check_trading_decision()
    print(strategy_chanlun_12h.direction_module_decision)
    print(strategy_chanlun_12h.pattern_module_decision)
    print(strategy_chanlun_12h.entry_module_decision)



