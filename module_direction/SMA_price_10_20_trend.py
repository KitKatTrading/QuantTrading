import pandas as pd
import talib

def main(df_OHLC_high, run_mode = 'live', num_candles=30, num_consecutive_candle=2):

    if run_mode == 'live':

        # Slice the DataFrame to the last 'num_candles' candles
        df_OHLC_high = df_OHLC_high.iloc[-num_candles:].copy()

        # Calculate SMAs
        df_OHLC_high['10_SMA'] = talib.SMA(df_OHLC_high['Close'], timeperiod=10)
        df_OHLC_high['20_SMA'] = talib.SMA(df_OHLC_high['Close'], timeperiod=20)

        # Check SMA conditions
        if all(df_OHLC_high['5_SMA'].iloc[-num_consecutive_candle:] > df_OHLC_high['10_SMA'].iloc[-num_consecutive_candle:]) \
                and all(df_OHLC_high['10_SMA'].iloc[-num_consecutive_candle:] > df_OHLC_high['20_SMA'].iloc[-num_consecutive_candle:]):
            return 1
        elif all(df_OHLC_high['5_SMA'].iloc[-num_consecutive_candle:] < df_OHLC_high['10_SMA'].iloc[-num_consecutive_candle:]) \
                and all(df_OHLC_high['10_SMA'].iloc[-num_consecutive_candle:] < df_OHLC_high['20_SMA'].iloc[-num_consecutive_candle:]):
            return -1
        else:
            return 0

    elif run_mode == 'backtest':

        # Calculate SMAs
        df_OHLC_high['10_SMA'] = talib.SMA(df_OHLC_high['Close'], timeperiod=10)
        df_OHLC_high['20_SMA'] = talib.SMA(df_OHLC_high['Close'], timeperiod=20)

        # List to store the qualified historical candles
        df_decision = pd.DataFrame(0, index=df_OHLC_high.index, columns=['decision'])

        # Iterate through the DataFrame
        for i in range(1, len(df_OHLC_high)):

            # if close price > 10SMA > 20SMA for the last num_consecutive_candle candles, then buy
            if df_OHLC_high['Close'].iloc[i] > df_OHLC_high['10_SMA'].iloc[i] > df_OHLC_high['20_SMA'].iloc[i]:
                df_decision['decision'].iloc[i] = 1

            # if close price < 10SMA < 20SMA for the last num_consecutive_candle candles, then sell
            elif df_OHLC_high['Close'].iloc[i] < df_OHLC_high['10_SMA'].iloc[i] < df_OHLC_high['20_SMA'].iloc[i]:
                df_decision['decision'].iloc[i] = -1

        df_decision.dropna(inplace=True)
        return df_decision
