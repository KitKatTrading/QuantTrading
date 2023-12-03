import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main(df_OHLC_low, RSI_overbought=65, RSI_oversold=35, run_mode='live'):

    # verify run_mode
    if run_mode != 'live':
        raise ValueError('Only live mode is supported')

    # initialize state variables
    RSI_extreme = 0
    RSI_extreme_trigger = False
    RSI_cross_21EMA = 0
    RSI_cross_21EMA_trigger = False
    RSI_value_cur = 50
    RSI_value_pre = 50

    # iterate through the dataframe
    for idx, RSI_value in df_OHLC_low['RSI'].iteritems():

        # update RSI_above_EMA status
        if RSI_value > df_OHLC_low['21_EMA'].loc[idx]:
            RSI_above_EMA_cur = True
        else:
            RSI_above_EMA_cur = False

        # if first iteration, set RSI_above_EMA_pre to RSI_above_EMA_cur and continue
        if idx == df_OHLC_low.index[0]:
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
            fig.add_trace(go.Scatter(x=df_OHLC_low.index, y=df_OHLC_low['21_EMA'], name='21 EMA'), row=2, col=1)

            if RSI_extreme == 1 and RSI_cross_21EMA == 1:
                return 1, fig, idx
            elif RSI_extreme == -1 and RSI_cross_21EMA == -1:
                return -1, fig, idx
            else:
                return 0, [], idx


