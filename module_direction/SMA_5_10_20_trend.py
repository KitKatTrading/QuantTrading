import talib

def main(pd_OHLC_high, num_candles=30, num_consecutive_candel=2):

    # Slice the DataFrame to the last 'num_candles' candles
    pd_OHLC_high = pd_OHLC_high.iloc[-num_candles:].copy()

    # Calculate SMAs
    pd_OHLC_high['5_SMA'] = talib.SMA(pd_OHLC_high['Close'], timeperiod=5)
    pd_OHLC_high['10_SMA'] = talib.SMA(pd_OHLC_high['Close'], timeperiod=10)
    pd_OHLC_high['20_SMA'] = talib.SMA(pd_OHLC_high['Close'], timeperiod=20)

    # Check SMA conditions
    if all(pd_OHLC_high['5_SMA'].iloc[-num_consecutive_candel:] > pd_OHLC_high['10_SMA'].iloc[-num_consecutive_candel:]) \
            and all(pd_OHLC_high['10_SMA'].iloc[-num_consecutive_candel:] > pd_OHLC_high['20_SMA'].iloc[-num_consecutive_candel:]):
        return 1
    elif all(pd_OHLC_high['5_SMA'].iloc[-num_consecutive_candel:] < pd_OHLC_high['10_SMA'].iloc[-num_consecutive_candel:]) \
            and all(pd_OHLC_high['10_SMA'].iloc[-num_consecutive_candel:] < pd_OHLC_high['20_SMA'].iloc[-num_consecutive_candel:]):
        return -1
    else:
        return 0
