import numpy as np
import plotly.graph_objs as go
from scipy.signal import find_peaks
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

DEBUG_PLOT = True
def apply_containing_pattern(df_OHLC_mid,
                             use_high_low=False,
                             debug_plot=DEBUG_PLOT,
                             ):

    ### --- Re-arrange the highs and lows
    # convert the 'Open' and 'Close' to a single series representing potential highs and lows
    if use_high_low:
        combined_highs = df_OHLC_mid[['High', 'Low']].max(axis=1)
        combined_lows = df_OHLC_mid[['High', 'Low']].min(axis=1)
    else:
        combined_highs = df_OHLC_mid[['Open', 'Close']].max(axis=1)
        combined_lows = df_OHLC_mid[['Open', 'Close']].min(axis=1)

    # combine the highs and lows into one dataframe df_HL, with the highs and lows as columns
    df_HL = pd.concat([combined_highs, combined_lows], axis=1)
    df_HL.columns = ['High', 'Low']

    # verify if the high is always higher than the low
    assert (df_HL['High'] >= df_HL['Low']).all()

    # create a numerical index saved as a column for later use
    df_HL['Idx'] = range(len(df_HL))

    ### --- Find the containing patterns
    # if any candle is contained by the previous candle (high < previous high and low > previous low), the
    # label the index of the candles that are contained by the previous candle
    counter_processing = 0
    while True:

        # verify if further processing is needed
        df_HL['contain_type_1'] = False  # contained by the previous candle
        df_HL['contain_type_2'] = False  # containing the previous candle
        df_HL.loc[(df_HL['High'] <= df_HL['High'].shift(1)) & (df_HL['Low'] >= df_HL['Low'].shift(1)), 'contain_type_1'] = True
        df_HL.loc[(df_HL['High'] >= df_HL['High'].shift(1)) & (df_HL['Low'] <= df_HL['Low'].shift(1)), 'contain_type_2'] = True
        df_HL['is_contained'] = df_HL['contain_type_1'] | df_HL['contain_type_2']

        # if there is no containing patterns, break the loop; otherwise, proceed to process the containing patterns
        if not df_HL['is_contained'].any():

            # visualization
            if debug_plot:
                fig = go.Figure(data=[go.Candlestick(x=df_HL.index,
                                                     open=df_HL['Low'],
                                                     high=df_HL['High'],
                                                     low=df_HL['Low'],
                                                     close=df_HL['High'],
                                                     name='HL')])
                fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
                fig.show()

            # drop temporary columns and break the loop
            df_HL = df_HL[['High', 'Low', 'Idx']]
            break

        # Print counter
        counter_processing += 1
        print(f'Processing containing patterns, round {counter_processing}')

        # Further processing starts here. First identify the first containing candle.
        df_HL['is_contained_to_process'] = False
        df_HL.loc[(df_HL['is_contained'] == True) & (
                    df_HL['is_contained'] != df_HL['is_contained'].shift(1)), 'is_contained_to_process'] = True

        # calculate the candle directions
        df_HL['direction'] = 'Bearish'  # Default to 'Bearish'
        df_HL.loc[df_HL['High'] > df_HL['High'].shift(1), 'direction'] = 'Bullish'

        # initialize the index to be removed
        df_HL['to_remove'] = False

        # --- Visualization
        if debug_plot:
            # plot the df_OHLC_mid
            fig = go.Figure(data=[go.Candlestick(x=df_HL.index,
                                                 open=df_HL['Low'],
                                                 high=df_HL['High'],
                                                 low=df_HL['Low'],
                                                 close=df_HL['High'],
                                                 name='HL')])

            # Add markers for 'is_contained'
            fig.add_trace(go.Scatter(x=df_HL[df_HL['is_contained']].index,
                                     y=df_HL[df_HL['is_contained']]['High'] + 1,
                                     mode='markers',
                                     marker=dict(color='blue', size=10),
                                     name='Contained'))

            # Add markers for 'is_contained_to_process'
            fig.add_trace(go.Scatter(x=df_HL[df_HL['is_contained_to_process']].index,
                                     y=df_HL[df_HL['is_contained_to_process']]['Low'] - 1,
                                     mode='markers',
                                     marker=dict(color='red', size=10),
                                     name='Contained to Process'))

            # Adjust layout to remove slide bar and add legend
            fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
            fig.show()

        # Process the first containing candles
        for i in range(1, len(df_HL)):
            if df_HL.iloc[i]['is_contained_to_process']:
                if df_HL.iloc[i]['direction'] == 'Bearish':
                    # Take the lower high and lower low of the current and previous highs/lows
                    df_HL.at[df_HL.index[i], 'High'] = min(df_HL.at[df_HL.index[i], 'High'],
                                                           df_HL.at[df_HL.index[i - 1], 'High'])
                    df_HL.at[df_HL.index[i], 'Low'] = min(df_HL.at[df_HL.index[i], 'Low'],
                                                          df_HL.at[df_HL.index[i - 1], 'Low'])
                    df_HL.at[df_HL.index[i - 1], 'to_remove'] = True
                else:
                    # Take the higher high and higher low of the current and previous highs/lows
                    df_HL.at[df_HL.index[i], 'High'] = max(df_HL.at[df_HL.index[i], 'High'],
                                                           df_HL.at[df_HL.index[i - 1], 'High'])
                    df_HL.at[df_HL.index[i], 'Low'] = max(df_HL.at[df_HL.index[i], 'Low'],
                                                          df_HL.at[df_HL.index[i - 1], 'Low'])
                    df_HL.at[df_HL.index[i - 1], 'to_remove'] = True

        # remove rows, reset status, and finish this round of processing
        df_HL = df_HL[df_HL['to_remove']==False]
        df_HL.drop(['is_contained', 'contain_type_1', 'contain_type_2', 'direction',
                    'is_contained_to_process', 'to_remove'], axis=1, inplace=True)

    return df_HL

def find_raw_peak_valley(df_HL, debug_plot=DEBUG_PLOT):

    # first find all potential peak and valley formations
    df_HL['is_peak'] = (df_HL['High'] > df_HL['High'].shift(1)) & (df_HL['High'] > df_HL['High'].shift(-1))
    df_HL['is_valley'] = (df_HL['Low'] < df_HL['Low'].shift(1)) & (df_HL['Low'] < df_HL['Low'].shift(-1))

    # calculate the direction - bullish or bearish
    # bullish - the current high is higher than the prior high
    # bearish - otherwise
    df_HL['direction'] = 'Bearish'  # Default to 'Bearish'
    df_HL.loc[df_HL['High'] > df_HL['High'].shift(1), 'direction'] = 'Bullish'

    # also calculate the peak and valley high/low values
    df_HL['pv_high'] = np.nan
    for i in range(1,len(df_HL)-1):
        if df_HL.iloc[i]['is_peak'] | df_HL.iloc[i]['is_valley']:

            # the pv_high is defined as the highest value among the current and previous and next highs
            df_HL.at[df_HL.index[i], 'pv_high'] = max(df_HL.at[df_HL.index[i], 'High'],
                                                      df_HL.at[df_HL.index[i - 1], 'High'],
                                                      df_HL.at[df_HL.index[i + 1], 'High'])
            df_HL.at[df_HL.index[i], 'pv_low'] = min(df_HL.at[df_HL.index[i], 'Low'],
                                                     df_HL.at[df_HL.index[i - 1], 'Low'],
                                                     df_HL.at[df_HL.index[i + 1], 'Low'])

    if debug_plot:
        # Create a new bar plot from 'Low' to 'High' for each bar
        fig = go.Figure()

        # Add bars for each direction
        fig.add_trace(go.Bar(x=df_HL.index,
                             y=df_HL['High'] - df_HL['Low'],
                             base=df_HL['Low'],
                             marker_color=df_HL['direction'].apply(lambda x: 'green' if x == 'Bullish' else 'red'),
                             name='Directional Bars'))

        # Add markers for 'is_contained'
        fig.add_trace(go.Scatter(x=df_HL[df_HL['is_peak']].index,
                                 y=df_HL[df_HL['is_peak']]['High'] + 1,
                                 mode='markers',
                                 marker=dict(color='blue', size=10),
                                 name='Peak'))

        # Add markers for 'is_contained_to_process'
        fig.add_trace(go.Scatter(x=df_HL[df_HL['is_valley']].index,
                                 y=df_HL[df_HL['is_valley']]['Low'] - 1,
                                 mode='markers',
                                 marker=dict(color='red', size=10),
                                 name='Valley'))

        # Adjust layout to remove slide bar and add legend
        fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
        fig.show()

    # save the peaks and valleys to a new DataFrame
    df_PV = df_HL[(df_HL['is_peak'] | df_HL['is_valley'])].copy()
    df_PV.loc[df_PV['is_peak']==True,'pv_type'] = 'Peak'
    df_PV.loc[df_PV['is_valley']==True,'pv_type'] = 'Valley'

    # verify that the peak and valley types are alternating without repeating
    assert (df_PV['pv_type'] != df_PV['pv_type'].shift(1)).all()

    return df_PV

def refine_peak_valley(df_PV, debug_plot=DEBUG_PLOT):

    # define status variables
    cur_peak = {}
    cur_valley = {}
    pre_peak = {}
    pre_valley = {}

    # initialize the status variables

    return 0


def main(df_OHLC_mid, num_candles=200,
         use_high_low=False):

    """ This mid-timeframe strategy is based on the Chan Theory and contains the following steps:
    Step 1. Process the OHLC data with containing patterns. (包含关系)
    Step 2. Find peaks and valleys (fractal tops and bottoms) (顶底分型以及线段)
    Step 3.

    """

    # slice the DataFrame to the last 'num_candles' candles
    df_OHLC_mid = df_OHLC_mid.iloc[-num_candles:]

    # Step 1 - Apply containing patterns
    df_HL = apply_containing_pattern(df_OHLC_mid, use_high_low=use_high_low)


    # Step 2 - Find peaks and valleys (fractal tops and bottoms)
    df_PV_raw = find_raw_peak_valley(df_HL)
    df_PV = refine_peak_valley(df_PV_raw)





    # Now df_peaks_valleys_combined is a DataFrame with the price, type, and original index (Date)

    # # --- Visualization
    # # Plot OHLC
    # fig = go.Figure(data=[go.Candlestick(x=df_OHLC_mid.index,
    #                                      open=df_OHLC_mid['Open'],
    #                                      high=df_OHLC_mid['High'],
    #                                      low=df_OHLC_mid['Low'],
    #                                      close=df_OHLC_mid['Close'],
    #                                      name='OHLC')])
    #
    # # Filter out peaks and valleys
    # peaks = df_pv_combined[df_pv_combined['Type'] == 'Peak']
    # valleys = df_pv_combined[df_pv_combined['Type'] == 'Valley']
    #
    # # Plot Peaks as scatter plot with larger markers
    # fig.add_trace(go.Scatter(x=peaks.index, y=peaks['Price'], mode='markers',
    #                          marker=dict(color='red', size=10), name='Peaks'))
    #
    # # Plot Valleys as scatter plot with larger markers
    # fig.add_trace(go.Scatter(x=valleys.index, y=valleys['Price'], mode='markers',
    #                          marker=dict(color='blue', size=10), name='Valleys'))
    #
    # # Add line plot connecting peaks and valleys
    # fig.add_trace(go.Scatter(x=df_pv_combined.index, y=df_pv_combined['Price'], mode='lines',
    #                          line=dict(color='black', dash='dash'), name='Peaks and Valleys'))
    #
    # # Adjust layout to remove slide bar and add legend
    # fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
    #
    # fig.show()

    return 0