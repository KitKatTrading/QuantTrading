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

    def high_timeframe_analysis(self):
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_high + '.csv')
        pd_OHLC_high = pd.read_csv(file_path, index_col=0)

        # Call the 'main' function from the strategy module
        self.direction_module_decision = self.strategy_high_timeframe.main(pd_OHLC_high)

    def mid_timeframe_analysis(self):
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
        pd_OHLC_mid = pd.read_csv(file_path, index_col=0)

        # Call the 'main' function from the strategy module
        self.pattern_module_decision = self.strategy_mid_timeframe.main(pd_OHLC_mid)

    def low_timeframe_analysis(self):
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
        pd_OHLC_mid = pd.read_csv(file_path, index_col=0)

        # Call the 'main' function from the strategy module
        self.entry_module_decision = self.strategy_low_timeframe.main(pd_OHLC_mid)


    def check_trading_decision(self):

        trading_decision = 0

        # directional module analysis
        self.high_timeframe_analysis()

        # only run pattern analysis if directional module analysis is not 0
        if self.direction_module_decision != 0:
            self.mid_timeframe_analysis()

            # only run trade entry module if both directional and pattern modules are meaningful:
            if self.pattern_module_decision * self.direction_module_decision == 1:
                self.low_timeframe_analysis()

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



