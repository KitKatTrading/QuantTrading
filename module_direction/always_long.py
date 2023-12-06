import pandas as pd
import talib

def main(df_OHLC_high, run_mode = 'live'):

    if run_mode == 'live':
        return 1

    elif run_mode == 'backtest':

        # List to store the qualified historical candles
        df_decision = pd.DataFrame(1, index=df_OHLC_high.index, columns=['decision'])

        return df_decision
