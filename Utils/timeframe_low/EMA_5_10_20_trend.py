import pandas as pd
import talib

def EMA_5_10_20_trend(pd_OHLC_high, num_candles=1000):
    # Slice the DataFrame to the last 'num_candles' candles
    pd_OHLC_high = pd_OHLC_high.iloc[-num_candles:]

    # Calculate EMAs
    pd_OHLC_high['5_EMA'] = talib.EMA(pd_OHLC_high['Close'], timeperiod=5)
    pd_OHLC_high['10_EMA'] = talib.EMA(pd_OHLC_high['Close'], timeperiod=10)
    pd_OHLC_high['20_EMA'] = talib.EMA(pd_OHLC_high['Close'], timeperiod=20)

    # Check EMA conditions
    if all(pd_OHLC_high['5_EMA'].iloc[-3:] > pd_OHLC_high['10_EMA'].iloc[-3:]) and all(pd_OHLC_high['10_EMA'].iloc[-3:] > pd_OHLC_high['20_EMA'].iloc[-3:]):
        return 1
    elif all(pd_OHLC_high['5_EMA'].iloc[-3:] < pd_OHLC_high['10_EMA'].iloc[-3:]) and all(pd_OHLC_high['10_EMA'].iloc[-3:] < pd_OHLC_high['20_EMA'].iloc[-3:]):
        return -1
    else:
        return 0
