import numpy as np
import talib
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from Utils.util_general import propagate_values

def main(df_OHLC_low,
         RSI_overbought=65, RSI_oversold=35,
         length_extreme_value_effective=12,
         run_mode='backtest'):

    # calculate indicators
    # Indicator #1: RSI and its EMA21
    df_OHLC_low['RSI'] = talib.RSI(df_OHLC_low['Close'], timeperiod=14)
    df_OHLC_low['RSI_EMA6'] = talib.EMA(df_OHLC_low['RSI'], timeperiod=6)
    df_OHLC_low['RSI_EMA12'] = talib.EMA(df_OHLC_low['RSI'], timeperiod=12)
    df_OHLC_low['RSI_EMA21'] = talib.EMA(df_OHLC_low['RSI'], timeperiod=21)
    df_OHLC_low.dropna(inplace=True)

    # verify run_mode
    if run_mode == 'live':

        # initialize state variables
        RSI_extreme = 0
        RSI_extreme_trigger = False
        RSI_cross_21EMA = 0
        RSI_cross_21EMA_trigger = False

        # iterate through the dataframe
        for idx, RSI_value in enumerate(df_OHLC_low['RSI']):

            RSI_21EMA_value = df_OHLC_low['RSI_EMA21'].iloc[idx]

            print(idx, RSI_value)
            # update RSI_above_EMA status
            if RSI_value > RSI_21EMA_value:
                RSI_above_EMA_cur = True
            else:
                RSI_above_EMA_cur = False

            # if first iteration, set RSI_above_EMA_pre to RSI_above_EMA_cur and continue
            if idx == 0:
                RSI_above_EMA_pre = RSI_above_EMA_cur
                continue

            # now check conditions
            if RSI_extreme_trigger is not True:
                # check for RSI extreme
                if RSI_value >= RSI_overbought:
                    RSI_extreme = -1
                    RSI_extreme_trigger = True
                elif RSI_value <= RSI_oversold:
                    RSI_extreme = 1
                    RSI_extreme_trigger = True

            # check for RSI cross 21 EMA
            if RSI_extreme_trigger and (not RSI_cross_21EMA_trigger):

                # overbought case, RSI cross 21 EMA from above
                if RSI_extreme == -1 and RSI_above_EMA_pre and (not RSI_above_EMA_cur):
                    RSI_cross_21EMA = -1
                    RSI_cross_21EMA_trigger = True

                # oversold case, RSI cross 21 EMA from below
                elif RSI_extreme == 1 and (not RSI_above_EMA_pre) and RSI_above_EMA_cur:
                    RSI_cross_21EMA = 1
                    RSI_cross_21EMA_trigger = True

            # check for entry condition
            if RSI_extreme_trigger and RSI_cross_21EMA_trigger:

                # visualization, draw the OHLC and RSI charts using plotly
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(x=df_OHLC_low.index, open=df_OHLC_low['Open'], high=df_OHLC_low['High'],
                                             low=df_OHLC_low['Low'], close=df_OHLC_low['Close'], name='OHLC'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_OHLC_low.index, y=df_OHLC_low['RSI'], name='RSI'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_OHLC_low.index, y=df_OHLC_low['RSI_EMA21'], name='21 EMA'), row=2, col=1)
                fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)

                if RSI_extreme == 1 and RSI_cross_21EMA == 1:
                    return 1, fig, idx
                elif RSI_extreme == -1 and RSI_cross_21EMA == -1:
                    return -1, fig, idx

            # update RSI_above_EMA_pre
            RSI_above_EMA_pre = RSI_above_EMA_cur

        # if at the end of the loop, no trade opportunity is found, return 0
        return 0, None, None

    # backtest mode to get all qualifying entries
    elif run_mode == 'backtest':

        ### first check overbough and oversold conditions and propagate the values
        # vectorize the values for overbought and oversold
        df_RSI_overbought = df_OHLC_low[['RSI']].copy()
        df_RSI_overbought['value'] = 0
        df_RSI_overbought.loc[df_RSI_overbought['RSI'] >= RSI_overbought, 'value'] = -1
        df_RSI_overbought.drop(columns=['RSI'], inplace=True)
        propagate_values(df_RSI_overbought, length_extreme_value_effective)
        df_OHLC_low['RSI_overbought'] = df_RSI_overbought['value']

        df_RSI_oversold = df_OHLC_low[['RSI']].copy()
        df_RSI_oversold['value'] = 0
        df_RSI_oversold.loc[df_RSI_oversold['RSI'] <= RSI_oversold, 'value'] = 1
        df_RSI_oversold.drop(columns=['RSI'], inplace=True)
        propagate_values(df_RSI_oversold, length_extreme_value_effective)
        df_OHLC_low['RSI_oversold'] = df_RSI_oversold['value']

        ### second check for death cross and golden cross
        # Create a column for death cross (current RSI < 21 EMA and previous RSI > 21 EMA)
        df_OHLC_low['death_cross'] = (df_OHLC_low['RSI_EMA6'] < df_OHLC_low['RSI_EMA12']) & \
                                     (df_OHLC_low['RSI_EMA6'].shift(1) > df_OHLC_low['RSI_EMA12'].shift(1))

        # Create a column for golden cross (current RSI > 21 EMA and previous RSI < 21 EMA)
        df_OHLC_low['golden_cross'] = (df_OHLC_low['RSI_EMA6'] > df_OHLC_low['RSI_EMA12']) & \
                                     (df_OHLC_low['RSI_EMA6'].shift(1) < df_OHLC_low['RSI_EMA12'].shift(1))

        # TODO - debug the sell case
        # Generate sell signals (-1) where there's an overbought condition and a death cross
        df_OHLC_low['decision_overbought_short'] = np.where((df_OHLC_low['RSI_overbought'] == -1) & df_OHLC_low['death_cross'], -1, 0)

        # Generate buy signals (1) where there's an oversold condition and a death cross
        df_OHLC_low['decision_oversold_long'] = np.where((df_OHLC_low['RSI_oversold'] == 1) & df_OHLC_low['golden_cross'], 1, 0)

        # Combine the signals into a single column
        df_OHLC_low['decision'] = df_OHLC_low['decision_overbought_short'] + df_OHLC_low['decision_oversold_long']

        ### clean up dataframe
        df_OHLC_low.drop(columns=['RSI_overbought', 'RSI_oversold',
                                  'decision_oversold_long', 'decision_overbought_short',
                                  'death_cross', 'golden_cross'], inplace=True)
        df_OHLC_low.dropna(inplace=True)

        # debug
        df_OHLC_low.to_csv('df_OHLC_low_with_state_variables.csv')

        return df_OHLC_low[['decision']]


