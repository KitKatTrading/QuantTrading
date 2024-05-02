import numpy as np
import talib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from utils.util_general import propagate_values

""" 
Price-action based entry module for the first new high/low strategy
- this function will generate a 1 or -1 when the current candle close makes the required number of times a new high or 
  low is made 
- only live mode is available
"""


def main(df_OHLC_low,
         anticipated_trade_direction,
         num_new_high_low_candles_for_trigger,
         run_mode='live'):

    # first verify run_mode
    if run_mode != 'live':
        raise ValueError("Only live mode is available for this module")

    # set the new high/low counter
    new_high_counter = 0
    new_low_counter = 0

    # iterate through the dataframe
    for idx, cur_close in enumerate(df_OHLC_low['Close']):

        # skip the first candle
        if idx == 0:
            continue

        # check the number of times new high or lows made
        if anticipated_trade_direction == "long":
            prev_high = df_OHLC_low['High'].iloc[idx - 1]
            if cur_close > prev_high:
                new_high_counter += 1
        elif anticipated_trade_direction == "short":
            prev_low = df_OHLC_low['Low'].iloc[idx - 1]
            if cur_close < prev_low:
                new_low_counter += 1

        # check if the number of new high/low candles is met
        if anticipated_trade_direction == "long":
            if new_high_counter == num_new_high_low_candles_for_trigger:
                return 1, df_OHLC_low.index[idx]
        elif anticipated_trade_direction == "short":
            if new_low_counter == num_new_high_low_candles_for_trigger:
                return -1, df_OHLC_low.index[idx]

    # if no trade is triggered, return 0
    return 0, None





