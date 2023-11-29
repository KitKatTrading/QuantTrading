import os
from datetime import datetime, timedelta
import pandas as pd
import importlib

from strategy import Strategy

# Adjust dir_data and add dir_utils to the Python path
current_script_dir = os.path.dirname(__file__)
dir_data = os.path.join(current_script_dir, '', 'module_data')
dir_data = os.path.normpath(dir_data)
dir_backtesting = os.path.join(current_script_dir, '', 'module_backtesting')


def convert_to_higher_timeframe(cur_date_low_timeframe, higher_timeframe):
    # Convert string to datetime object
    datetime_obj = datetime.strptime(cur_date_low_timeframe, '%Y-%m-%d %H:%M:%S%z')

    if higher_timeframe == "1d":
        # For daily timeframe, extract the date
        converted_datetime = datetime_obj.date()
    elif higher_timeframe == "12h":
        # For 12h timeframe, round to the nearest 12:00 or 00:00
        if datetime_obj.hour < 12:
            converted_datetime = datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            converted_datetime = datetime_obj.replace(hour=12, minute=0, second=0, microsecond=0)
    elif higher_timeframe == "1w":
        # For weekly timeframe, find the start of the week (Monday)
        weekday = datetime_obj.weekday()
        start_of_week = datetime_obj - timedelta(days=weekday)
        converted_datetime = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    elif higher_timeframe in ["8h", "4h"]:
        # For 8h or 4h timeframes, find the start of the nearest period
        hours = 8 if higher_timeframe == "8h" else 4
        hour_rounded_down = datetime_obj.hour - (datetime_obj.hour % hours)
        converted_datetime = datetime_obj.replace(hour=hour_rounded_down, minute=0, second=0, microsecond=0)
    else:
        raise ValueError("Unsupported higher timeframe")

    return converted_datetime




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

        # Set backtesting directory - use the current date as the backtesting directory name
        self.backtesting_dir = os.path.join(dir_backtesting, name_strategy, name_symbol, datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.backtesting_dir)

        # Define the strategy object
        self.strategy = Strategy(name_symbol=self.name_symbol, data_source=self.data_source, name_strategy=self.name_strategy,
                                 timeframe_high=self.timeframe_high, timeframe_mid=self.timeframe_mid, timeframe_low=self.timeframe_low,
                                 function_high_timeframe=self.function_high_timeframe,
                                 function_mid_timeframe=self.function_mid_timeframe,
                                 function_low_timeframe=self.function_low_timeframe)

    def run_backtesting_iterative(self):

        # Load the OHLC data for the high timeframe directional module
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_high + '.csv')
        df_OHLC_high = pd.read_csv(file_path, index_col=0)
        df_OHLC_high = df_OHLC_high.loc[self.bt_start_date: self.bt_end_date]

        # Load the OHLC data for the mid timeframe pattern module
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
        df_OHLC_mid = pd.read_csv(file_path, index_col=0)
        df_OHLC_mid = df_OHLC_mid.loc[self.bt_start_date: self.bt_end_date]

        # Load the OHLC data for the low timeframe entry module
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_low + '.csv')
        df_OHLC_low = pd.read_csv(file_path, index_col=0)
        df_OHLC_low = df_OHLC_low.loc[self.bt_start_date: self.bt_end_date]

        # Run the direction module
        df_decision_direction = self.strategy.strategy_high_timeframe.main(df_OHLC_high, run_mode='backtest')
        df_decision_direction_long = df_decision_direction.loc[df_decision_direction['value'] == 1]
        df_decision_direction_short = df_decision_direction.loc[df_decision_direction['value'] == -1]

        # The pattern module can't be vectorized - will need to iterate through the DataFrame

        # Run the entry module
        df_decision_entry = self.strategy.strategy_low_timeframe.main(df_OHLC_low, run_mode='backtest')
        df_decision_entry_long = df_decision_entry.loc[df_decision_entry['value'] == 1]
        df_decision_entry_short = df_decision_entry.loc[df_decision_entry['value'] == -1]

        # First check long trades
        # loop through the entry module, and verify direction and pattern modules
        for rows, columns in df_decision_entry_long.iterrows():
            print(rows, columns['value'])

            # get the current date
            cur_date_low_timeframe = rows

            # convert the low timeframe date to the open time of the corresponding high timeframe candle
            cur_date_high_timeframe = convert_to_higher_timeframe(cur_date_low_timeframe, self.timeframe_high)
            cur_date_high_timeframe = cur_date_high_timeframe.strftime('%Y-%m-%d %H:%M:%S+00:00')

            if cur_date_high_timeframe not in df_decision_direction_long.index:
                continue
            else:
                # now check the pattern module, first convert the date to the mid timeframe
                cur_date_mid_timeframe = convert_to_higher_timeframe(cur_date_low_timeframe, self.timeframe_mid)

                # get rid of the open candle
                if self.timeframe_mid == "12h":
                    hours_offset = 12
                    cur_date_mid_timeframe = cur_date_mid_timeframe - timedelta(hours=hours_offset)
                    cur_date_mid_timeframe = cur_date_mid_timeframe.strftime('%Y-%m-%d %H:%M:%S+00:00')

                # now check the pattern module
                df_OHLC_mid_temp = df_OHLC_mid.loc[:cur_date_mid_timeframe]
                pattern_module_decision, fig_hubs = self.strategy.strategy_mid_timeframe.main(df_OHLC_mid_temp,
                                                                                              debug_plot=False,
                                                                                              num_candles=300)

                if pattern_module_decision == 1:
                    print('Long trading opportunity')
                    trading_decision = 1
                    #### Trading module ####
                else:
                    print('Pattern module not satisfied')





        # for rows, columns in df_decision_direction.iterrows():
        #     if columns['value'] == 0:
        #         print('Buy')


        # # # Run the pattern module
        #
        # df_decision_setup = self.strategy.strategy_mid_timeframe.main(df_OHLC_mid, run_mode='backtest')







if __name__ == '__main__':

    # Define the backtesting object
    backtesting = Backtesting(name_symbol='BTCUSDT', data_source='binance', name_strategy='chanlun_12h',
                              timeframe_high='1w', timeframe_mid='12h', timeframe_low='1h',
                              function_high_timeframe='SMA_5_10_20_trend',
                              function_mid_timeframe='chanlun_central_hub',
                              function_low_timeframe='RSI_divergence',
                              bt_start_date='2020-06-01 00:00:00+00:00',
                              bt_end_date='2023-11-01 00:00:00+00:00')

    # Run the backtesting
    trading_decision = backtesting.run_backtesting_iterative()

    print(trading_decision)

    print('Done!')