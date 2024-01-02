import pandas as pd
import talib

def main(df_OHLC_high, run_mode = 'live', num_consecutive_candle=2):

    if run_mode == 'live':
        return 0

    elif run_mode == 'backtest':

        # Calculate SMAs
        df_OHLC_high['SMA10'] = talib.SMA(df_OHLC_high['Close'], timeperiod=10)
        df_OHLC_high['SMA20'] = talib.SMA(df_OHLC_high['Close'], timeperiod=20)

        # List to store the qualified historical candles
        df_decision = pd.DataFrame(0, index=df_OHLC_high.index, columns=['decision'])

        # Iterate through the DataFrame
        for i in range(num_consecutive_candle, len(df_OHLC_high)):

            # two conditions must be met for the last num_consecutive_candle candles to be qualified:
            # 1. close price > SMA10 > SMA20
            # 2. SMA10 must be trending (up or down) for the last num_consecutive_candle candles

            df_OHLC_high['Close-SMA10'] = df_OHLC_high['Close'] - df_OHLC_high['SMA10']
            df_OHLC_high['SMA10-SMA20'] = df_OHLC_high['SMA10'] - df_OHLC_high['SMA20']
            df_OHLC_high['SMA10_trend'] = df_OHLC_high['SMA10'] - df_OHLC_high['SMA10'].shift(1)

            # Check SMA conditions for the last num_consecutive_candle candles
            if all(df_OHLC_high['Close-SMA10'].iloc[i - num_consecutive_candle:i] > 0) and \
                all(df_OHLC_high['SMA10-SMA20'].iloc[i - num_consecutive_candle:i] > 0) and \
                all(df_OHLC_high['SMA10_trend'].iloc[i - num_consecutive_candle:i] > 0):
                df_decision['decision'].iloc[i] = 1
            elif all(df_OHLC_high['Close-SMA10'].iloc[i - num_consecutive_candle:i] < 0) and \
                    all(df_OHLC_high['SMA10-SMA20'].iloc[i - num_consecutive_candle:i] < 0) and \
                    all(df_OHLC_high['SMA10_trend'].iloc[i - num_consecutive_candle:i] < 0):
                df_decision['decision'].iloc[i] = -1

        df_decision.dropna(inplace=True)
        return df_decision
