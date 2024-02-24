import numpy as np
import talib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from Utils.util_general import propagate_values

""" 
RSI-based pinbar entry strategy
- First check for overbought and oversold conditions
- Then check the pinbar condition
- Generate sell signals (-1) where there's an overbought condition and a long lower pinbar
- Generate buy signals (1) where there's an oversold condition and a long upper pinbar
- Combine the signals into a single column
- Return the dataframe with the decision column
"""


def main(df_OHLC_low,
         RSI_period=14,
         RSI_overbought=60,
         RSI_oversold=30,
         Vol_MA_period=200,
         Vol_MA_thres_multiplier=1,
         ATR_period=200,
         pinbar_body_ATR_thres_multiplier=0.8,
         num_candles_to_plot=50,
         run_mode='live'):

    # calculate indicators
    # Indicator #1: RSI, ATR, and average volume
    df_OHLC_low['RSI'] = talib.RSI(df_OHLC_low['Close'], timeperiod=RSI_period)
    df_OHLC_low['ATR'] = talib.ATR(df_OHLC_low['High'], df_OHLC_low['Low'], df_OHLC_low['Close'], timeperiod=ATR_period)
    df_OHLC_low['Volume_MA'] = talib.MA(df_OHLC_low['Volume'], timeperiod=Vol_MA_period)

    # calculate the upper and lower pinbar length
    # for lower pinbar, calculate the lower of the open and close vs the low
    df_OHLC_low['lower_pinbar_length'] = np.where(df_OHLC_low['Open'] > df_OHLC_low['Close'],
                                                    df_OHLC_low['Close'] - df_OHLC_low['Low'],
                                                    df_OHLC_low['Open'] - df_OHLC_low['Low'])
    # for upper pinbar, calculate the higher of the open and close vs the high
    df_OHLC_low['upper_pinbar_length'] = np.where(df_OHLC_low['Open'] < df_OHLC_low['Close'],
                                                    df_OHLC_low['High'] - df_OHLC_low['Close'],
                                                    df_OHLC_low['High'] - df_OHLC_low['Open'])

    df_OHLC_low.dropna(inplace=True)

    # verify run_mode
    if run_mode == 'live':

        decision_entry = 0

        # get the current parameter values
        RSI_cur = df_OHLC_low['RSI'].iloc[-1]
        ATR_cur = df_OHLC_low['ATR'].iloc[-1]
        Vol_cur = df_OHLC_low['Volume'].iloc[-1]
        Vol_MA_cur = df_OHLC_low['Volume_MA'].iloc[-1]
        lower_pinbar_cur = df_OHLC_low['lower_pinbar_length'].iloc[-1]
        upper_pinbar_cur = df_OHLC_low['upper_pinbar_length'].iloc[-1]

        # check for entry signal
        # long case
        if RSI_cur <= RSI_oversold and \
            lower_pinbar_cur > pinbar_body_ATR_thres_multiplier * ATR_cur and \
            Vol_cur > Vol_MA_thres_multiplier * Vol_MA_cur:
            decision_entry = 1

        # short case
        elif RSI_cur >= RSI_overbought and \
            upper_pinbar_cur > pinbar_body_ATR_thres_multiplier * ATR_cur and \
            Vol_cur > Vol_MA_thres_multiplier * Vol_MA_cur:
            decision_entry = -1

        # make plots if there's a trade opportunity
        if decision_entry != 0:

            ### make plot using plotly,
            df_OHLC_plot = df_OHLC_low.iloc[-num_candles_to_plot:]
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

            # set the first panel height to be 1/2 of the total height
            fig.update_layout(height=600, width=900)


            # remove slider
            fig.update_layout(xaxis_rangeslider_visible=False)

            ### OHLC
            fig.add_trace(go.Candlestick(x=df_OHLC_plot.index,
                                         open=df_OHLC_plot['Open'],
                                         high=df_OHLC_plot['High'],
                                         low=df_OHLC_plot['Low'],
                                         close=df_OHLC_plot['Close'],
                                         name='Candlestick'), row=1, col=1)

            ### RSI
            # plot RSI as columns, set the range from 0 to 100, set the colum to light yellow
            fig.add_trace(go.Bar(x=df_OHLC_plot.index, y=df_OHLC_plot['RSI'], name='RSI'), row=2, col=1)
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            fig.update_traces(marker_color='lightyellow', marker_line_color='black', marker_line_width=1.5, opacity=0.6, row=2, col=1)

            # add a horizontal line at the overbought and oversold levels
            fig.add_shape(type="line", x0=df_OHLC_plot.index[0], y0=RSI_overbought, x1=df_OHLC_plot.index[-1], y1=RSI_overbought,
                            line=dict(color="red", width=1), row=2, col=1)
            fig.add_shape(type="line", x0=df_OHLC_plot.index[0], y0=RSI_oversold, x1=df_OHLC_plot.index[-1], y1=RSI_oversold,
                            line=dict(color="green", width=1), row=2, col=1)

            ### Volume
            fig.add_trace(go.Bar(x=df_OHLC_plot.index, y=df_OHLC_plot['Volume'], name='Volume'), row=3, col=1)

            # plot the volume MA
            fig.add_trace(go.Scatter
                            (x=df_OHLC_plot.index, y=df_OHLC_plot['Volume_MA'], name='Volume_MA'),
                            row=3, col=1)

            ### Postprocessing
            # put the figure legend on the middle and top
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            # use larger fonts
            fig.update_layout(font=dict(size=18))


            return decision_entry, fig
        else:
            return decision_entry, None



    # backtest mode to get all qualifying entries
    # elif run_mode == 'backtest':
    #
    #
    #
    #     # TODO - debug the sell case
    #     # Generate sell signals (-1) where there's an overbought condition and a death cross
    #     df_OHLC_low['decision_overbought_short'] = np.where((df_OHLC_low['RSI_overbought'] == -1) & df_OHLC_low['death_cross'], -1, 0)
    #
    #     # Generate buy signals (1) where there's an oversold condition and a death cross
    #     df_OHLC_low['decision_oversold_long'] = np.where((df_OHLC_low['RSI_oversold'] == 1) & df_OHLC_low['golden_cross'], 1, 0)
    #
    #     # Combine the signals into a single column
    #     df_OHLC_low['decision'] = df_OHLC_low['decision_overbought_short'] + df_OHLC_low['decision_oversold_long']
    #
    #     ### clean up dataframe
    #     df_OHLC_low.drop(columns=['RSI_overbought', 'RSI_oversold',
    #                               'decision_oversold_long', 'decision_overbought_short',
    #                               'death_cross', 'golden_cross'], inplace=True)
    #     df_OHLC_low.dropna(inplace=True)
    #
    #     # debug
    #     df_OHLC_low.to_csv('df_OHLC_low_with_state_variables.csv')
    #
    #     return df_OHLC_low[['decision']]


