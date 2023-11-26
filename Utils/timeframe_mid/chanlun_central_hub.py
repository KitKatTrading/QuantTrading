import numpy as np
import plotly.graph_objs as go
from scipy.signal import find_peaks
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

DEBUG_PLOT = True
DEBUG_PRINT = True

def debug_plot_peak_valley_factors(df_PV, df_HL):
    fig = go.Figure()

    # Add bars for each direction
    fig.add_trace(go.Bar(x=df_HL.index,
                         y=df_HL['High'] - df_HL['Low'],
                         base=df_HL['Low'],
                         marker_color=df_HL['direction'].apply(
                             lambda x: 'green' if x == 'Bullish' else 'red'),
                         name='Directional Bars'))

    # Add markers for 'peak'
    fig.add_trace(go.Scatter(x=df_PV[df_PV['pv_type'] == 'Peak'].index,
                             y=df_PV[df_PV['pv_type'] == 'Peak']['High'] * 1.01,
                             mode='markers',
                             marker=dict(color='blue', size=10),
                             name='Peak'))

    # Add markers for 'valley'
    fig.add_trace(go.Scatter(x=df_PV[df_PV['pv_type'] == 'Valley'].index,
                             y=df_PV[df_PV['pv_type'] == 'Valley']['Low'] * 0.99,
                             mode='markers',
                             marker=dict(color='red', size=10),
                             name='Valley'))

    # Adjust layout to remove slide bar and add legend
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
    fig.show()


def apply_containing_pattern(df_OHLC_mid,
                             use_high_low=False,
                             debug_plot=False,
                             debug_print=False,
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

def find_raw_peaks_valleys(df_HL, debug_plot=DEBUG_PLOT):

    # first find all potential peak and valley factors
    df_HL['is_peak'] = (df_HL['High'] > df_HL['High'].shift(1)) & (df_HL['High'] > df_HL['High'].shift(-1))
    df_HL['is_valley'] = (df_HL['Low'] < df_HL['Low'].shift(1)) & (df_HL['Low'] < df_HL['Low'].shift(-1))

    # calculate the direction - bullish or bearish
    # bullish - the current high is higher than the prior high
    # bearish - otherwise
    df_HL['direction'] = 'Bearish'  # Default to 'Bearish'
    df_HL.loc[df_HL['High'] > df_HL['High'].shift(1), 'direction'] = 'Bullish'

    # also calculate the peak and valley high/low values
    df_HL['pv_high'] = np.nan
    for i in range(1, len(df_HL)-1):
        if df_HL.iloc[i]['is_peak'] | df_HL.iloc[i]['is_valley']:

            # the pv_high is defined as the highest value among the current and previous and next highs
            df_HL.at[df_HL.index[i], 'pv_high'] = max(df_HL.at[df_HL.index[i], 'High'],
                                                      df_HL.at[df_HL.index[i - 1], 'High'],
                                                      df_HL.at[df_HL.index[i + 1], 'High'])
            df_HL.at[df_HL.index[i], 'pv_low'] = min(df_HL.at[df_HL.index[i], 'Low'],
                                                     df_HL.at[df_HL.index[i - 1], 'Low'],
                                                     df_HL.at[df_HL.index[i + 1], 'Low'])

    # save the peaks and valleys to a new DataFrame
    df_PV = df_HL[(df_HL['is_peak'] | df_HL['is_valley'])].copy()
    df_PV.loc[df_PV['is_peak']==True, 'pv_type'] = 'Peak'
    df_PV.loc[df_PV['is_valley']==True, 'pv_type'] = 'Valley'

    # manually add the first and last factors as factors
    df_PV_start = df_HL.iloc[[0]].copy()
    df_PV_start['pv_high'] = df_PV_start['High']
    df_PV_start['pv_low'] = df_PV_start['Low']
    if df_PV['pv_type'].iloc[0] == 'Peak':
        df_PV_start['pv_type'] = 'Valley'
        df_PV_start['is_valley'] = True
    else:
        df_PV_start['pv_type'] = 'Peak'
        df_PV_start['is_peak'] = True

    df_PV_end = df_HL.iloc[[-1]].copy()
    df_PV_end['pv_high'] = df_PV_end['High']
    df_PV_end['pv_low'] = df_PV_end['Low']
    if df_PV['pv_type'].iloc[-1] == 'Peak':
        df_PV_end['pv_type'] = 'Valley'
        df_PV_end['is_valley'] = True
    else:
        df_PV_end['pv_type'] = 'Peak'
        df_PV_end['is_peak'] = True

    df_PV = pd.concat([df_PV_start, df_PV, df_PV_end], axis=0)

    # verify that the peak and valley types are alternating without repeating
    assert (df_PV['pv_type'] != df_PV['pv_type'].shift(1)).all()

    return df_PV


def refine_peaks_valleys(df_PV, df_HL, debug_plot=DEBUG_PLOT):

    while True:

        ### Extract clusters that have several too-close factors
        clusters = []
        cluster_start = None
        for i in range(0, len(df_PV)):
            print(f'Processing {i}th factor')

            # if the current factor is the first factor
            if i == 0:
                if df_PV['Idx'].iloc[i + 1] - df_PV['Idx'].iloc[i] < 4:
                    if cluster_start is None:
                        cluster_start = i

            # if the current factor is the last factor
            elif i == len(df_PV) - 1:
                if df_PV['Idx'].iloc[i] - df_PV['Idx'].iloc[i - 1] < 4:
                    if cluster_start is not None:
                        # End of the current cluster
                        cluster_end = i
                        clusters.append((cluster_start, cluster_end))
                        cluster_start = None

            # general case when the factor is in the middle
            else:
                if df_PV['Idx'].iloc[i] - df_PV['Idx'].iloc[i - 1] >= 4 and \
                      df_PV['Idx'].iloc[i + 1] - df_PV['Idx'].iloc[i] < 4:
                        if cluster_start is None:
                            cluster_start = i  # Start of a new cluster
                elif df_PV['Idx'].iloc[i] - df_PV['Idx'].iloc[i - 1] < 4 and \
                      df_PV['Idx'].iloc[i + 1] - df_PV['Idx'].iloc[i] >= 4:
                        if cluster_start is not None:
                            # End of the current cluster
                            cluster_end = i
                            clusters.append((cluster_start, cluster_end))
                            cluster_start = None  # Reset for the next cluster

        # if there is no too-close pv factors, break the loop
        if len(clusters) == 0:
            if debug_plot:
                debug_plot_peak_valley_factors(df_PV, df_HL)
            break

        # --- Process one by one factor in each cluster
        for cluster in clusters:
            


        # --- Visualization
        if debug_plot:
            debug_plot_peak_valley_factors(df_PV, df_HL)




    # # # print the clusters and their indices
    # # print(f'Clusters: {clusters}')
    # # for cluster in clusters:
    # #     print(f'Processing cluster: {cluster}')
    # #     print(f'Index cluster: {df_PV.iloc[cluster[0]:cluster[1]+1].index.tolist()}')
    #
    # # separate the clusters into even and odd clusters
    # clusters_even = [cluster for cluster in clusters if (cluster[1] - cluster[0] + 1) % 2 == 0]
    # clusters_odd = [cluster for cluster in clusters if (cluster[1] - cluster[0] + 1) % 2 == 1]
    #
    # ### Process the all clusters
    # index_to_remove = []
    #
    # ### process the even clusters first
    # for cluster in clusters_even:
    #
    #     # print the cluster
    #     print(f'Processing cluster: {cluster}')
    #     print(f'Index cluster: {df_PV.iloc[cluster[0]:cluster[1]+1].index.tolist()}')
    #
    #     # find the cluster high and lows
    #     cluster_df = df_PV.iloc[cluster[0]:cluster[1] + 1]
    #     cluster_high = cluster_df['High'].max()
    #     cluster_high_idx = cluster_df['High'].idxmax()
    #     cluster_low = cluster_df['Low'].min()
    #     cluster_low_idx = cluster_df['Low'].idxmin()
    #
    #     # if the first factor is a peak, then check if the cluster low is lower than the prior factor's low
    #     if cluster_df.iloc[0]['pv_type'] == 'Peak':
    #         # if the cluster low is lower than the prior factor's low, then only keep cluster_low_idx
    #         if cluster_low < df_PV.iloc[cluster[0]-1]['Low']:
    #             index_to_remove += cluster_df.index.tolist()
    #             index_to_remove.append(df_PV.index[cluster[0]-1])
    #             index_to_remove.remove(cluster_low_idx)
    #         # otherwise, remove all factors in the cluster
    #         else:
    #             index_to_remove += cluster_df.index.tolist()
    #     # if the first factor is a valley, then check if the cluster high is higher than the prior factor's high
    #     else:
    #         # if the cluster high is higher than the prior factor's high, then only keep cluster_high_idx
    #         if cluster_high > df_PV.iloc[cluster[0]-1]['High']:
    #             index_to_remove += cluster_df.index.tolist()
    #             index_to_remove.append(df_PV.index[cluster[0]-1])
    #             index_to_remove.remove(cluster_high_idx)
    #         # otherwise, remove all factors in the cluster
    #         else:
    #             index_to_remove += cluster_df.index.tolist()
    #
    #
    #
    # ### now process the odd clusters
    # for cluster in clusters_odd:
    #
    #     # print the cluster
    #     print(f'Processing cluster: {cluster}')
    #     print(f'Index cluster: {df_PV.iloc[cluster[0]:cluster[1]+1].index.tolist()}')
    #
    #     # find the cluster high and lows
    #     cluster_df = df_PV.iloc[cluster[0]:cluster[1] + 1]
    #     cluster_high = cluster_df['High'].max()
    #     cluster_high_idx = cluster_df['High'].idxmax()
    #     cluster_low = cluster_df['Low'].min()
    #     cluster_low_idx = cluster_df['Low'].idxmin()
    #
    #     # if the first factor is a peak, then all we need is just one representative peak
    #     if cluster_df.iloc[0]['pv_type'] == 'Peak':
    #         index_to_remove += cluster_df.index.tolist()
    #         index_to_remove.remove(cluster_high_idx)
    #     # if the first factor is a valley, then all we need is just one representative valley
    #     else:
    #         index_to_remove += cluster_df.index.tolist()
    #         index_to_remove.remove(cluster_low_idx)
    #
    # # remove all identified factors
    # df_PV.drop(index_to_remove, inplace=True)
    #
    # # assert the factors are in alternating order
    # assert (df_PV['pv_type'] != df_PV['pv_type'].shift(1)).all()
    #
    # # visualization
    # if debug_plot:
    #     # Create a new bar plot from 'Low' to 'High' for each bar
    #     fig = go.Figure()
    #
    #     # Add bars for each direction
    #     fig.add_trace(go.Bar(x=df_HL.index,
    #                          y=df_HL['High'] - df_HL['Low'],
    #                          base=df_HL['Low'],
    #                          marker_color=df_HL['direction'].apply(
    #                              lambda x: 'green' if x == 'Bullish' else 'red'),
    #                          name='Directional Bars'))
    #
    #     # Add markers for 'is_contained'
    #     fig.add_trace(go.Scatter(x=df_PV[df_PV['pv_type'] == 'Peak'].index,
    #                              y=df_PV[df_PV['pv_type'] == 'Peak']['High'] * 1.01,
    #                              mode='markers',
    #                              marker=dict(color='blue', size=10),
    #                              name='Peak'))
    #
    #     # Add markers for 'is_contained_to_process'
    #     fig.add_trace(go.Scatter(x=df_PV[df_PV['pv_type'] == 'Valley'].index,
    #                              y=df_PV[df_PV['pv_type'] == 'Valley']['Low'] * 0.99,
    #                              mode='markers',
    #                              marker=dict(color='red', size=10),
    #                              name='Valley'))
    #
    #     # Adjust layout to remove slide bar and add legend
    #     fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
    #     fig.show()

    return 0


    # # --- pre-processing special case where there are too close pv factors
    # counter_processing_single = 0
    # while True:
    #     counter_processing_single += 1
    #     print(f'Processing too-close pv factors, round {counter_processing_single}')
    #
    #     # Find pv factors that are too close to the previous one but distant enough from the next one
    #     df_PV['is_too_close'] = (((df_PV['Idx'] - df_PV['Idx'].shift(1)) >= 4) &      ### current - previous
    #                              ((df_PV['Idx'].shift(-1) - df_PV['Idx']) < 4))     ### next - current
    #     if len(df_PV) > 0:
    #         df_PV.at[df_PV.index[0], 'is_too_close'] = (df_PV['Idx'].iloc[1] - df_PV['Idx'].iloc[0]) < 4
    #
    #     # if there is no too-close pv factors, drop temprarily columns and break the loop
    #     if not df_PV['is_too_close'].any():
    #         df_PV.drop(['is_too_close'], axis=1, inplace=True)
    #         break
    #
    #     # process the leading round of ending too-close pv factors
    #     idx_to_remove = []
    #     for i in range(1, len(df_PV)):
    #         if df_PV.iloc[i]['is_too_close']:
    #             # if the too-close factor is a peak
    #             if df_PV.iloc[i]['pv_type'] == 'Peak':
    #                 # if the ending is a peak, then the next valley is the new valley
    #                 next_valley = df_PV.at[df_PV.index[i + 1], 'pv_low']
    #                 pre_valley = df_PV.at[df_PV.index[i - 1], 'pv_low']
    #                 if next_valley < pre_valley:
    #                     idx_to_remove.append([df_PV.index[i-1], df_PV.index[i]])
    #                 else:
    #                     idx_to_remove.append([df_PV.index[i], df_PV.index[i+1]])
    #             # if the too-close factor is a valley
    #             else:
    #                 # if the ending is a valley, then the next peak is the new peak
    #                 next_peak = df_PV.at[df_PV.index[i + 1], 'pv_high']
    #                 pre_peak = df_PV.at[df_PV.index[i - 1], 'pv_high']
    #                 if next_peak > pre_peak:
    #                     idx_to_remove.append([df_PV.index[i-1], df_PV.index[i]])
    #                 else:
    #                     idx_to_remove.append([df_PV.index[i], df_PV.index[i+1]])
    #
    #     # remove the too-close pv factors
    #     for idx in idx_to_remove:
    #         df_PV.drop(idx, inplace=True)
    #


    # # define state variable templates
    # pv_template = {
    #     'idx': -1,  # value1 can be of any data type, e.g., int, float, string, etc.
    #     'high': -1.0,  # value2 can be different from value1's type
    #     'low': -1.0,  # and so on...
    # }
    #
    # line_template = {
    #     'idx_start': -1,
    #     'idx_end': -1,
    #     'direction': None,
    # }
    #
    # # initialize the state variables
    # cur_peak = pv_template.copy()
    # cur_valley = pv_template.copy()
    # pre_peak = pv_template.copy()
    # pre_valley = pv_template.copy()
    # cur_line = line_template.copy()
    #
    # pre_peak['idx'] = 0
    # pre_valley['idx'] = 0
    # cur_line['idx_start'] = 0
    # cur_line['idx_end'] = df_PV.iloc[0]['Idx']
    #
    # # if the first factor is a peak, then the first line is going up
    # if df_PV.iloc[0]['pv_type'] == 'Peak':
    #     cur_valley['idx'] = 0
    #     cur_valley['high'] = 1e-10
    #     cur_valley['low'] = 1e-10
    #     cur_peak['idx'] = df_PV.iloc[0]['Idx']
    #     cur_peak['high'] = df_PV.iloc[0]['pv_high']
    #     cur_peak['low'] = df_PV.iloc[0]['pv_low']
    #     cur_line['direction'] = 'up'
    # # if the first factor is a valley, then the first line is going down
    # else:
    #     cur_peak['idx'] = 0
    #     cur_peak['high'] = 9e9
    #     cur_peak['low'] = 9e9
    #     cur_valley['idx'] = df_PV.iloc[0]['Idx']
    #     cur_valley['high'] = df_PV.iloc[0]['pv_high']
    #     cur_valley['low'] = df_PV.iloc[0]['pv_low']
    #     cur_line['direction'] = 'down'
    #
    # # loop through all factors to validate lines and pvs
    # num_changes = 1
    # while num_changes > 0:
    #     num_changes = 0
    #     if cur_line['direction'] == 'down':
    #         next_first = 'peak'
    #         next_second = 'valley'
    #     else:
    #         next_first = 'valley'
    #         next_second = 'peak'
    #
    #


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
    df_PV_raw = find_raw_peaks_valleys(df_HL)
    df_PV = refine_peaks_valleys(df_PV_raw, df_HL)





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