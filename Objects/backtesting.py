import os
import pandas as pd
import talib
from datetime import datetime, timedelta
from Utils.util_general import debug_logging
from Objects.strategy import Strategy

# Adjust dir_data and add dir_utils to the Python path
# current_script_dir = os.path.dirname(__file__)
# dir_data = os.path.join(current_script_dir, '', '../module_data')
# dir_data = os.path.normpath(dir_data)
# # dir_backtesting = os.path.join(current_script_dir, '', '../module_backtesting')
dir_data = 'module_data'
dir_backtesting = 'module_backtesting'

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
    elif higher_timeframe in ["8h", "4h", "2h", "1h"]:
        # For 8h, 4h, 2h or 1h timeframes, find the start of the nearest period
        hours = int(higher_timeframe[0])
        hour_rounded_down = datetime_obj.hour - (datetime_obj.hour % hours)
        converted_datetime = datetime_obj.replace(hour=hour_rounded_down, minute=0, second=0, microsecond=0)
    else:
        raise ValueError("Unsupported higher timeframe")

    return converted_datetime

def check_single_trade_outcome(df_OHLC_low, entry_datetime, entry_price, direction, exit_datetime, exit_price):
    """ Check the outcome of a single trade """
    # Get the entry and exit prices
    entry_price = df_OHLC_low['Close'].loc[entry_datetime]
    exit_price = df_OHLC_low['Close'].loc[exit_datetime]

    # Calculate the PnL
    if direction == 'long':
        pnl = exit_price - entry_price
    elif direction == 'short':
        pnl = entry_price - exit_price
    else:
        raise ValueError("Unsupported direction")

    return pnl

class OneTrade:
    """ A class to represent a single trade """
    def __init__(self, symbol, name_strategy, entry_datetime, entry_price, exit_datetime, exit_price, direction, pnl):
        self.symbol = symbol
        self.name_strategy = name_strategy
        self.entry_datetime = entry_datetime
        self.entry_price = entry_price
        self.exit_datetime = exit_datetime
        self.exit_price = exit_price
        self.direction = direction
        self.pnl = pnl

    def __str__(self):
        return f"entry_datetime = {self.entry_datetime}, entry_price = {self.entry_price}, exit_datetime = {self.exit_datetime}, exit_price = {self.exit_price}, direction = {self.direction}, pnl = {self.pnl}"


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

    def run_backtesting_vectorize_high_low(self,
                                           manual_review_each_trade=True,
                                           trade_direction='long',
                                           save_plots=True,
                                           save_csv=True,
                                           ):
        """ Run the backtesting using vectorized high and low timeframe modules """

        ### SECTION 1 - Preprocessing the entry signals that can be vectorized
        ### DIRECTION MODULE
        # Load data
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_high + '.csv')
        df_OHLC_high = pd.read_csv(file_path, index_col=0)
        df_OHLC_high = df_OHLC_high.loc[self.bt_start_date: self.bt_end_date]

        # Vectorize the high timeframe directional module decisions
        df_decision_direction = self.strategy.strategy_high_timeframe.main(df_OHLC_high, run_mode='backtest')
        df_decision_direction_long = df_decision_direction.loc[df_decision_direction['decision'] == 1]
        df_decision_direction_short = df_decision_direction.loc[df_decision_direction['decision'] == -1]

        ### PATTERN MODULE
        # Load data
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_mid + '.csv')
        df_OHLC_mid = pd.read_csv(file_path, index_col=0)
        df_OHLC_mid = df_OHLC_mid.loc[self.bt_start_date: self.bt_end_date]

        ### ENTRY MODULE
        # Load data
        file_path = os.path.join(self.data_dir, self.name_symbol + '_' + self.timeframe_low + '.csv')
        df_OHLC_low = pd.read_csv(file_path, index_col=0)
        df_OHLC_low = df_OHLC_low.loc[self.bt_start_date: self.bt_end_date]

        # Vectorize the low timeframe entry module decisions
        df_decision_entry = self.strategy.strategy_low_timeframe.main(df_OHLC_low, run_mode='backtest')
        df_decision_entry_long = df_decision_entry.loc[df_decision_entry['decision'] == 1]
        df_decision_entry_short = df_decision_entry.loc[df_decision_entry['decision'] == -1]

        ### Output
        if save_csv:
            # Save the csv for debugging
            df_decision_entry.to_csv(os.path.join(self.backtesting_dir, 'decision_entry.csv'))
            df_decision_entry_long.to_csv(os.path.join(self.backtesting_dir, 'df_decision_entry_long.csv'))
            df_decision_entry_short.to_csv(os.path.join(self.backtesting_dir, 'df_decision_entry_short.csv'))

        ### ------------ Iterations for backtesting ------------ ###
        """ Since we have vectorized the entry module, we can now just loop through the identified entries exclusively"""
        # Initialize the entry log
        df_entry_log_long = pd.DataFrame(0, index=df_OHLC_low.index,
                                    columns=['decision_direction', 'decision_pattern', 'decision_entry', 'decision_final'])


        ### SECTION 2 - Loop through the entries to execute the trades

        # initialize state variables
        cooling_down_counter = 0
        max_cooling_down = 12

        # for idx_low, datetime_low in enumerate(df_decision_entry_long.index):
        for idx_low, datetime_low in enumerate(df_OHLC_low.index):

            # update the cooling down counter
            if cooling_down_counter > 0:
                cooling_down_counter -= 1

            ### ENTRY MODULE
            # get the current date and entry decision
            cur_date_low_timeframe = datetime_low
            decision_entry = df_decision_entry['decision'].loc[cur_date_low_timeframe]
            df_entry_log_long['decision_entry'].loc[cur_date_low_timeframe] = decision_entry
            print(f"low_timeframe = {cur_date_low_timeframe}")

            ### DIRECTION MODULE
            # convert datetime
            cur_date_high_timeframe = convert_to_higher_timeframe(cur_date_low_timeframe, self.timeframe_high)
            cur_date_high_timeframe = cur_date_high_timeframe.strftime('%Y-%m-%d %H:%M:%S+00:00')

            # read direction module
            try:
                decision_direction = df_decision_direction['decision'].loc[cur_date_high_timeframe]
                df_entry_log_long['decision_direction'].loc[cur_date_low_timeframe] = decision_direction
            except:
                print(f"- high_timeframe = {cur_date_high_timeframe}")
                print('- error - invalid direction module')
                continue

            ### PATTERN MODULE
            # convert dataframe
            cur_date_mid_timeframe = convert_to_higher_timeframe(cur_date_low_timeframe, self.timeframe_mid)
            if self.timeframe_mid == "12h":
                hours_offset_mid = 12
            elif self.timeframe_mid == '8h':
                hours_offset_mid = 8
            elif self.timeframe_mid == '4h':
                hours_offset_mid = 4
            elif self.timeframe_mid == '2h':
                hours_offset_mid = 2
            elif self.timeframe_mid == '1h':
                hours_offset_mid = 1
            cur_date_mid_timeframe = cur_date_mid_timeframe - timedelta(hours=hours_offset_mid)
            cur_date_mid_timeframe = cur_date_mid_timeframe.strftime('%Y-%m-%d %H:%M:%S+00:00')

            # run specific pattern strategy
            df_OHLC_mid_temp = df_OHLC_mid.loc[:cur_date_mid_timeframe].copy()
            df_OHLC_mid_temp = df_OHLC_mid.copy()
            try:
                decision_pattern, fig_hubs = (
                    self.strategy.strategy_mid_timeframe.main(
                        df_OHLC_mid_temp,
                        name_symbol=self.name_symbol,
                        time_frame=self.timeframe_mid,
                        num_candles=300,
                    ))
                # fig_hubs.show()
                df_entry_log_long['decision_pattern'].loc[cur_date_low_timeframe] = decision_pattern
            except:
                print('- error - invalid pattern module')
                continue