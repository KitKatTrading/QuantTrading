import os
from datetime import datetime, timedelta
import talib
import pandas as pd
# import importlib

from utils import *
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

    def run_backtesting_vectorize_high_low(self):

        # initialize the cooling down period
        cooling_down_period = 12
        cooling_down_counter = 0

        ### ------------ Process high timeframe data ------------ ###
        # Load data
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_high + '.csv')
        df_OHLC_high = pd.read_csv(file_path, index_col=0)
        df_OHLC_high = df_OHLC_high.loc[self.bt_start_date: self.bt_end_date]

        # Vectorize the high timeframe directional module decisions
        df_decision_direction = self.strategy.strategy_high_timeframe.main(df_OHLC_high, run_mode='backtest')
        df_decision_direction_long = df_decision_direction.loc[df_decision_direction['decision'] == 1]
        df_decision_direction_short = df_decision_direction.loc[df_decision_direction['decision'] == -1]

        ### ------------ Process mid timeframe data ------------ ###
        # Load data
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
        df_OHLC_mid = pd.read_csv(file_path, index_col=0)
        df_OHLC_mid = df_OHLC_mid.loc[self.bt_start_date: self.bt_end_date]

        ### ------------ Process low timeframe data ------------ ###
        # Load data
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_low + '.csv')
        df_OHLC_low = pd.read_csv(file_path, index_col=0)
        df_OHLC_low = df_OHLC_low.loc[self.bt_start_date: self.bt_end_date]

        # Customize indicators
        # Indicator #1: RSI and its EMA21
        df_OHLC_low['RSI'] = talib.RSI(df_OHLC_low['Close'], timeperiod=14)
        df_OHLC_low['RSI_EMA6'] = talib.EMA(df_OHLC_low['RSI'], timeperiod=6)
        df_OHLC_low['RSI_EMA12'] = talib.EMA(df_OHLC_low['RSI'], timeperiod=12)
        df_OHLC_low['RSI_EMA24'] = talib.EMA(df_OHLC_low['RSI'], timeperiod=24)


        df_OHLC_low.dropna(inplace=True)

        # Vectorize the low timeframe entry module decisions
        df_decision_entry = self.strategy.strategy_low_timeframe.main(df_OHLC_low, run_mode='backtest')
        df_decision_entry_long = df_decision_entry.loc[df_decision_entry['decision'] == 1]
        df_decision_entry_short = df_decision_entry.loc[df_decision_entry['decision'] == -1]

        # Save the csv for debugging
        df_decision_entry.to_csv(os.path.join(self.backtesting_dir, 'decision_entry.csv'))
        df_decision_entry_long.to_csv(os.path.join(self.backtesting_dir, 'df_decision_entry_long.csv'))
        df_decision_entry_short.to_csv(os.path.join(self.backtesting_dir, 'df_decision_entry_short.csv'))

        ### ------------ Initialize entry log ------------ ###
        # Initialize the entry log
        df_entry_log_long = pd.DataFrame(0, index=df_OHLC_low.index,
                                    columns=['decision_direction', 'decision_pattern', 'decision_entry', 'decision_final'])

        ### ------------ Iterations ------------ ###
        """ Since we have vectorized the entry module, we can just loop through the identified entries"""

        ### ------------ Long entries ------------ ###
        # for idx_low, datetime_low in enumerate(df_decision_entry_long.index):
        for idx_low, datetime_low in enumerate(df_OHLC_low.index):

            # update the cooling down counter
            if cooling_down_counter > 0:
                cooling_down_counter -= 1

            # get the current date
            cur_date_low_timeframe = datetime_low
            decision_entry = df_decision_entry['decision'].loc[cur_date_low_timeframe]
            df_entry_log_long['decision_entry'].loc[cur_date_low_timeframe] = decision_entry
            print(f"low_timeframe = {cur_date_low_timeframe}")

            # debug get all decisions
            cur_date_high_timeframe = convert_to_higher_timeframe(cur_date_low_timeframe, self.timeframe_high)
            cur_date_high_timeframe = cur_date_high_timeframe.strftime('%Y-%m-%d %H:%M:%S+00:00')

            try:
                decision_direction = df_decision_direction['decision'].loc[cur_date_high_timeframe]
                df_entry_log_long['decision_direction'].loc[cur_date_low_timeframe] = decision_direction
            except:
                print(f"- high_timeframe = {cur_date_high_timeframe}")
                print('- error - direction module not satisfied')
                continue

            # pattern direction
            cur_date_mid_timeframe = convert_to_higher_timeframe(cur_date_low_timeframe, self.timeframe_mid)
            if self.timeframe_mid == "12h":
                hours_offset_mid = 12
            elif self.timeframe_mid == '4h':
                hours_offset_mid = 4
            cur_date_mid_timeframe = cur_date_mid_timeframe - timedelta(hours=hours_offset_mid)
            cur_date_mid_timeframe = cur_date_mid_timeframe.strftime('%Y-%m-%d %H:%M:%S+00:00')

            # check the pattern module
            df_OHLC_mid_temp = df_OHLC_mid.loc[:cur_date_mid_timeframe].copy()
            df_OHLC_mid_temp = df_OHLC_mid.copy()
            try:
                decision_pattern, fig_hubs = (
                    self.strategy.strategy_mid_timeframe.main(df_OHLC_mid_temp))
                df_entry_log_long['decision_pattern'].loc[cur_date_low_timeframe] = decision_pattern
            except:
                print('- error - pattern module not satisfied')
                continue

            # ### ------------ Direction module processing------------ ###
            #
            # # convert the low timeframe date to the open time of the corresponding high timeframe candle
            # cur_date_high_timeframe = convert_to_higher_timeframe(cur_date_low_timeframe, self.timeframe_high)
            # cur_date_high_timeframe = cur_date_high_timeframe.strftime('%Y-%m-%d %H:%M:%S+00:00')
            #
            # # get the decision from the direction module
            # try:
            #     decision_direction = df_decision_direction['decision'].loc[cur_date_high_timeframe]
            #     df_entry_log_long['decision_direction'].loc[cur_date_low_timeframe] = decision_direction
            # except:
            #     print(f"- high_timeframe = {cur_date_high_timeframe}")
            #     print('- error - direction module not satisfied')
            #     continue
            #
            # # skip if decision_direction is not the same as the decision_entry
            # if decision_direction != decision_entry:
            #     continue
            # else:
            #     print(f"- high_timeframe = {cur_date_high_timeframe}")
            #
            # ### ------------ Pattern module processing------------ ###
            # # get the datetime for the mid timeframe and get rid of the open candle (for real time processing)
            # cur_date_mid_timeframe = convert_to_higher_timeframe(cur_date_low_timeframe, self.timeframe_mid)
            # if self.timeframe_mid == "12h":
            #     hours_offset_mid = 12
            # elif self.timeframe_mid == '4h':
            #     hours_offset_mid = 4
            # cur_date_mid_timeframe = cur_date_mid_timeframe - timedelta(hours=hours_offset_mid)
            # cur_date_mid_timeframe = cur_date_mid_timeframe.strftime('%Y-%m-%d %H:%M:%S+00:00')
            #
            # # check the pattern module
            # df_OHLC_mid_temp = df_OHLC_mid.loc[:cur_date_mid_timeframe].copy()
            # try:
            #     decision_pattern, fig_hubs = (
            #         self.strategy.strategy_mid_timeframe.main(df_OHLC_mid_temp, num_candles=500))
            #     df_entry_log_long['decision_pattern'].loc[cur_date_low_timeframe] = decision_pattern
            # except:
            #     print('- error - pattern module not satisfied')
            #     continue
            #
            # if decision_pattern == decision_entry:
            #     if cooling_down_counter > 0:
            #         continue
            #     elif cooling_down_counter == 0:
            #         cooling_down_counter = cooling_down_period
            #         print(f"-- mid_timeframe = {cur_date_mid_timeframe}")
            #         print('-- identified trading setup')
            #         df_entry_log_long['decision_final'].loc[cur_date_low_timeframe] = \
            #             decision_pattern * decision_entry * decision_direction
            #         fig_hubs.show()
            # else:
            #     print('-- pattern module not satisfied')

            ### ------------ Entry module processing------------ ###

        # save the entry log
        df_entry_log_long.to_csv(os.path.join(self.backtesting_dir, 'entry_log_long.csv'))

if __name__ == '__main__':

    # Define the backtesting object
    backtesting = Backtesting(name_symbol='BTCUSDT', data_source='binance', name_strategy='chanlun',
                              timeframe_high='1w', timeframe_mid='12h', timeframe_low='1h',
                              function_high_timeframe='always_long',
                              function_mid_timeframe='chanlun',
                              function_low_timeframe='RSI_extreme_cross',
                              bt_start_date='2023-01-01 00:00:00+00:00',
                              bt_end_date='2023-11-01 00:00:00+00:00')

    # Run the backtesting
    trading_decision = backtesting.run_backtesting_vectorize_high_low()

    debug_logging(trading_decision)

