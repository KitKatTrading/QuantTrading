import numpy as np
import pandas as pd
import plotly.graph_objs as go
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

DEBUG_PLOT = False
DEBUG_PRINT = False

def debug_logging(message, debug_print=DEBUG_PRINT):
    if debug_print:
        print(message)

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

    # Add lines connecting peaks and valleys
    for i in range(len(df_PV) - 1):
        current_row = df_PV.iloc[i]
        next_row = df_PV.iloc[i + 1]

        current_value = current_row['High'] if current_row['pv_type'] == 'Peak' else current_row['Low']
        next_value = next_row['High'] if next_row['pv_type'] == 'Peak' else next_row['Low']

        fig.add_trace(go.Scatter(x=[current_row.name, next_row.name],
                                 y=[current_value, next_value],
                                 mode='lines',
                                 line=dict(color='black', width=1),
                                 showlegend=False))  # Don't show these lines in the legend

    # Adjust layout to remove slide bar and add legend
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
    fig.show()

def debug_plot_segments(df_PV, df_HL, df_segments):
    # Create a new figure
    fig = go.Figure()

    # Add bars for each direction in df_HL
    fig.add_trace(go.Bar(x=df_HL.index,
                         y=df_HL['High'] - df_HL['Low'],
                         base=df_HL['Low'],
                         marker_color=df_HL['direction'].apply(
                             lambda x: 'green' if x == 'Bullish' else 'red'),
                         name='Directional Bars',
                         opacity=0.6))  # Adjusting opacity for better visualization

    # Iterate over each segment
    for i, row in enumerate(df_segments.iterrows()):
        # Get start and end indexes
        _, row = row  # row is a tuple of (index, row_data)
        start_idx = row['idx_start']
        end_idx = row['idx_end']

        # Find the corresponding rows in df_PV
        start_row = df_PV.loc[start_idx]
        end_row = df_PV.loc[end_idx]

        # Determine which value to use (High for Peak and Low for Valley)
        start_value = start_row['High'] if start_row['pv_type'] == 'Peak' else start_row['Low']
        end_value = end_row['High'] if end_row['pv_type'] == 'Peak' else end_row['Low']

        # Add a line trace for this segment
        fig.add_trace(go.Scatter(x=[start_idx, end_idx],
                                 y=[start_value, end_value],
                                 mode='lines+markers',
                                 line=dict(width=2, color='black'),
                                 name='Segment' if i == 0 else None,  # Only show legend for the first segment
                                 showlegend=i == 0))

    # Set layout options
    fig.update_layout(title='Line Segments',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      showlegend=True,
                      xaxis_rangeslider_visible=False)

    # Show the plot
    fig.show()

def debug_plot_hubs_using_HL(df_PV, df_HL, df_hubs):

    # Create a new figure
    fig = go.Figure()

    # calculate the hub directions
    pre_hub_high = -1.
    pre_hub_low = -1.
    df_hubs['color'] = 'green'
    for idx_hub, hub in df_hubs.iterrows():
        cur_hub_high = hub['high']
        cur_hub_low = hub['low']
        if cur_hub_low > pre_hub_high:
            df_hubs.at[idx_hub, 'direction'] = 'up'
            df_hubs.at[idx_hub, 'color'] = 'green'
        elif cur_hub_high < pre_hub_low:
            df_hubs.at[idx_hub, 'direction'] = 'down'
            df_hubs.at[idx_hub, 'color'] = 'red'
        # update pre hub high and low
        pre_hub_high = cur_hub_high
        pre_hub_low = cur_hub_low

    # Add bars for each direction in df_HL
    fig.add_trace(go.Bar(x=df_HL.index,
                         y=df_HL['High'] - df_HL['Low'],
                         base=df_HL['Low'],
                         marker_color=df_HL['direction'].apply(
                             lambda x: 'green' if x == 'Bullish' else 'red'),
                         name='Directional Bars',
                         opacity=0.6))  # Adjusting opacity for better visualization

    # Iterate over df_PV to draw consecutive lines
    for i in range(len(df_PV) - 1):
        # Current point
        start_idx = df_PV.index[i]
        start_value = df_PV.iloc[i]['factor_value']

        # Next point
        end_idx = df_PV.index[i + 1]
        end_value = df_PV.iloc[i + 1]['factor_value']

        # Add a line trace for this segment
        fig.add_trace(go.Scatter(x=[start_idx, end_idx],
                                 y=[start_value, end_value],
                                 mode='lines+markers',
                                 line=dict(width=2, color='black')))

    # Iterate over each hub to draw rectangles
    for _, hub in df_hubs.iterrows():
        hub_color = hub['color']

        # Add a rectangle to represent the hub zone
        fig.add_shape(type="rect",
                      x0=hub['start_idx'], y0=hub['low'],
                      x1=hub['end_idx'], y1=hub['high'],
                      line=dict(color=hub_color),
                      fillcolor=hub_color,
                      opacity=0.2)

    # Set layout options
    fig.update_layout(title='Line Segments with Hubs',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      showlegend=False,
                      xaxis_rangeslider_visible=False)

    # Show the plot
    fig.show()

def debug_plot_hubs_using_OHLC(df_PV, df_OHLC, df_hubs):
    # Create a new figure
    fig = go.Figure()

    # Calculate the hub directions
    pre_hub_high = -1.
    pre_hub_low = -1.
    df_hubs['color'] = 'green'
    for idx_hub, hub in df_hubs.iterrows():
        cur_hub_high = hub['high']
        cur_hub_low = hub['low']
        if cur_hub_low > pre_hub_high:
            df_hubs.at[idx_hub, 'direction'] = 'up'
            df_hubs.at[idx_hub, 'color'] = 'green'
        elif cur_hub_high < pre_hub_low:
            df_hubs.at[idx_hub, 'direction'] = 'down'
            df_hubs.at[idx_hub, 'color'] = 'red'
        pre_hub_high = cur_hub_high
        pre_hub_low = cur_hub_low

    # Add candlestick trace for OHLC data
    fig.add_trace(go.Candlestick(x=df_OHLC.index,
                                 open=df_OHLC['Open'],
                                 high=df_OHLC['High'],
                                 low=df_OHLC['Low'],
                                 close=df_OHLC['Close'],
                                 name='OHLC'))

    # Iterate over df_PV to draw consecutive lines
    for i in range(len(df_PV) - 1):
        # Current point
        start_idx = df_PV.index[i]
        start_value = df_PV.iloc[i]['factor_value']

        # Next point
        end_idx = df_PV.index[i + 1]
        end_value = df_PV.iloc[i + 1]['factor_value']

        # Add a line trace for this segment
        fig.add_trace(go.Scatter(x=[start_idx, end_idx],
                                 y=[start_value, end_value],
                                 mode='lines+markers',
                                 line=dict(width=2, color='black')))

    # Iterate over each hub to draw rectangles
    for _, hub in df_hubs.iterrows():
        hub_color = hub['color']

        # Add a rectangle to represent the hub zone
        fig.add_shape(type="rect",
                      x0=hub['start_idx'], y0=hub['low'],
                      x1=hub['end_idx'], y1=hub['high'],
                      line=dict(color=hub_color),
                      fillcolor=hub_color,
                      opacity=0.2)

    # Set layout options
    fig.update_layout(title='OHLC with Line Segments and Hubs',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      showlegend=True,
                      xaxis_rangeslider_visible=False)

    # Show the plot
    fig.show()

def debug_plot_hubs(df_PV, df_HL, df_OHLC, df_hubs):
    # Create a new figure
    fig = go.Figure()

    # Calculate the hub directions
    pre_hub_high = -1.
    pre_hub_low = -1.
    df_hubs['color'] = 'green'
    for idx_hub, hub in df_hubs.iterrows():
        cur_hub_high = hub['high']
        cur_hub_low = hub['low']
        if cur_hub_low > pre_hub_high:
            df_hubs.at[idx_hub, 'direction'] = 'up'
            df_hubs.at[idx_hub, 'color'] = 'green'
        elif cur_hub_high < pre_hub_low:
            df_hubs.at[idx_hub, 'direction'] = 'down'
            df_hubs.at[idx_hub, 'color'] = 'red'
        pre_hub_high = cur_hub_high
        pre_hub_low = cur_hub_low

    # Add HL bars
    fig.add_trace(go.Bar(x=df_HL.index,
                         y=df_HL['High'] - df_HL['Low'],
                         base=df_HL['Low'],
                         marker_color=df_HL['direction'].apply(
                             lambda x: 'green' if x == 'Bullish' else 'red'),
                         name='HL Bars',
                         opacity=0.6))

    # Add OHLC candlestick trace
    fig.add_trace(go.Candlestick(x=df_OHLC.index,
                                 open=df_OHLC['Open'],
                                 high=df_OHLC['High'],
                                 low=df_OHLC['Low'],
                                 close=df_OHLC['Close'],
                                 name='OHLC',
                                 increasing_line=dict(color='green'),
                                 decreasing_line=dict(color='red')))

    # Iterate over df_PV to draw consecutive lines
    for i in range(len(df_PV) - 1):
        # Current point
        start_idx = df_PV.index[i]
        start_value = df_PV.iloc[i]['factor_value']

        # Next point
        end_idx = df_PV.index[i + 1]
        end_value = df_PV.iloc[i + 1]['factor_value']

        # Add a line trace for this segment
        fig.add_trace(go.Scatter(x=[start_idx, end_idx],
                                 y=[start_value, end_value],
                                 mode='lines+markers',
                                 line=dict(width=2, color='black'),
                                 name='PV Lines' if i == 0 else None,  # Only show legend for the first line
                                 showlegend=(i == 0)))

    # Iterate over each hub to draw rectangles
    for _, hub in df_hubs.iterrows():
        hub_color = hub['color']

        # Add a rectangle to represent the hub zone
        fig.add_shape(type="rect",
                      x0=hub['start_idx'], y0=hub['low'],
                      x1=hub['end_idx'], y1=hub['high'],
                      line=dict(color=hub_color),
                      fillcolor=hub_color,
                      opacity=0.2)

    # Set layout options
    fig.update_layout(title='Chan Theory Central Zones (Hubs)',
                      # xaxis_title='Date',
                      # yaxis_title='Price',
                      showlegend=True,
                      legend=dict(orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"),
                      xaxis_rangeslider_visible=False)

    # Show the plot
    fig.show()

    return fig

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
        debug_logging(f'Processing containing patterns, round {counter_processing}')

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
    df_PV['pv_type'] = ''
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

    if debug_plot:
        debug_plot_peak_valley_factors(df_PV, df_HL)

    while True:

        ### Extract clusters that have several too-close factors
        clusters = []
        cluster_start = None
        for i in range(0, len(df_PV)):
            debug_logging(f'Processing {i}th factor')

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
        index_to_remove = []
        cluster = clusters[0]
        debug_logging(f'Processing cluster: {cluster}')
        debug_logging(f'Index cluster: {df_PV.iloc[cluster[0]:cluster[1]+1].index.tolist()}')

        # if the first factor is a peak
        if df_PV.iloc[cluster[0]]['pv_type'] == 'Peak':

            # -- extract neighboring factors
            # current peak
            cur_peak = df_PV.iloc[cluster[0]]['High']
            cur_peak_idx = df_PV.index[cluster[0]]

            # current valley (the one before)
            if cluster[0] == 0:
                cur_valley = -1.0
            else:
                cur_valley = df_PV.iloc[cluster[0]-1]['Low']
                cur_valley_idx = df_PV.index[cluster[0]-1]

            # next valley
            next_valley = df_PV.iloc[cluster[0]+1]['Low']
            next_valley_idx = df_PV.index[cluster[0]+1]

            # next peak
            if cluster[0] == len(df_PV)-2:
                next_peak = 9e9
            else:
                next_peak = df_PV.iloc[cluster[0]+2]['High']
                next_peak_idx = df_PV.index[cluster[0]+2]

            # --- process the cluster
            if next_valley < cur_valley:
                # if the next valley is lower than the current valley, then the next valley is the new valley
                index_to_remove += [cur_peak_idx, cur_valley_idx]
            elif next_peak > cur_peak:
                # if the next peak is higher than the current peak, then the next peak is the new peak
                index_to_remove += [cur_peak_idx, next_valley_idx]
            else:
                # if the next peak is lower than the current peak, then the current peak is the new peak
                index_to_remove += [next_peak_idx, next_valley_idx]

        # if the first factor is a valley
        else:

            # -- extract neighboring factors
            # current valley
            cur_valley = df_PV.iloc[cluster[0]]['Low']
            cur_valley_idx = df_PV.index[cluster[0]]

            # current peak (the one before)
            if cluster[0] == 0:
                cur_peak = 9e9
            else:
                cur_peak = df_PV.iloc[cluster[0]-1]['High']
                cur_peak_idx = df_PV.index[cluster[0]-1]

            # next peak
            next_peak = df_PV.iloc[cluster[0]+1]['High']
            next_peak_idx = df_PV.index[cluster[0]+1]

            # next valley
            if cluster[0] == len(df_PV)-2:
                next_valley = -1.0
            else:
                next_valley = df_PV.iloc[cluster[0]+2]['Low']
                next_valley_idx = df_PV.index[cluster[0]+2]

            # --- process the cluster
            if next_peak > cur_peak:
                # if the next peak is higher than the current peak, then the next peak is the new peak
                index_to_remove += [cur_valley_idx, cur_peak_idx]
            elif next_valley < cur_valley:
                # if the next valley is lower than the current valley, then the next valley is the new valley
                index_to_remove += [cur_valley_idx, next_peak_idx]
            else:
                # if the next valley is higher than the current valley, then the current valley is the new valley
                index_to_remove += [next_valley_idx, next_peak_idx]

        # remove all identified factors
        df_PV.drop(index_to_remove, inplace=True)

        # --- Visualization
        if debug_plot:
            debug_plot_peak_valley_factors(df_PV, df_HL)

    # assert the factors are in alternating order
    assert (df_PV['pv_type'] != df_PV['pv_type'].shift(1)).all()

    return df_PV

def find_segments(df_PV, df_HL, debug_plot=DEBUG_PLOT):

    # assert the alternating pattern
    assert (df_PV['pv_type'] != df_PV['pv_type'].shift(1)).all()

    if debug_plot:
        debug_plot_peak_valley_factors(df_PV, df_HL)

    # Define an empty list to store segment data
    segments = []

    # extract segments using a loop
    df_PV_copy = df_PV.copy()
    while True:
        seg_start = df_PV_copy.index[0]

        if seg_start == df_PV.index[-1]:
            break

        # if the starting factor is a valley
        if df_PV_copy.iloc[0]['pv_type'] == 'Valley':
            seg_direction = 'up'

            # check if a line just becomes a segment
            segment_failed = False

            # case when there are only two factors left, then it is the last "segment"
            if len(df_PV_copy) == 2:
                segment_failed = True
                seg_end = df_PV_copy.index[1]
            elif len(df_PV_copy) == 3:
                segment_failed = True
                if df_PV_copy.iloc[2]['Low'] < df_PV_copy.iloc[0]['Low']:
                    seg_end = df_PV_copy.index[2]
                else:
                    seg_end = df_PV_copy.index[1]

            if segment_failed:
                segment_data = {'idx_start': seg_start, 'idx_end': seg_end, 'direction': seg_direction}
                segments.append(segment_data)
            # otherwise, a minimum segment can be found for at least two consecutive valleys
            else:
                # increasing trend
                for i in range(3, len(df_PV_copy)):

                    # case when the segment hits a first valley that breaks the monotonous increasing trend
                    if df_PV_copy.iloc[i]['pv_type'] == 'Valley':
                        if df_PV_copy.iloc[i]['Low'] < df_PV_copy.iloc[i-2]['Low']:
                            seg_end = df_PV_copy.index[i-1]
                            segment_data = {'idx_start': seg_start, 'idx_end': seg_end, 'direction': seg_direction}
                            segments.append(segment_data)
                            break
                    # case when the segment hits a first peak that breaks the monotonous increasing trend
                    else:
                        if df_PV_copy.iloc[i]['High'] < df_PV_copy.iloc[i-2]['High']:
                            seg_end = df_PV_copy.index[i-2]
                            segment_data = {'idx_start': seg_start, 'idx_end': seg_end, 'direction': seg_direction}
                            segments.append(segment_data)
                            break

                    # case when it is the end of the df_PV_copy
                    if df_PV_copy.index[i] == df_PV.index[-1]:
                        seg_end = df_PV_copy.index[-1]
                        segment_data = {'idx_start': seg_start, 'idx_end': seg_end, 'direction': seg_direction}
                        segments.append(segment_data)
                        break

        # if the starting factor is a peak
        else:
            seg_direction = 'down'

            # check if a line just becomes a segment
            segment_failed = False

            # case when there are only two factors left, then it is the last "segment"
            if len(df_PV_copy) == 2:
                segment_failed = True
                seg_end = df_PV_copy.index[1]
            elif len(df_PV_copy) == 3:
                segment_failed = True
                if df_PV_copy.iloc[2]['High'] > df_PV_copy.iloc[0]['High']:
                    seg_end = df_PV_copy.index[2]
                else:
                    seg_end = df_PV_copy.index[1]

            if segment_failed:
                segment_data = {'idx_start': seg_start, 'idx_end': seg_end, 'direction': seg_direction}
                segments.append(segment_data)
            # otherwise, a minimum segment can be found for at least two consecutive valleys
            else:
                # decreasing trend
                for i in range(3, len(df_PV_copy)):

                    # case when the segment hits a first peak that breaks the monotonous decreasing trend
                    if df_PV_copy.iloc[i]['pv_type'] == 'Peak':
                        if df_PV_copy.iloc[i]['High'] > df_PV_copy.iloc[i-2]['High']:
                            seg_end = df_PV_copy.index[i-1]
                            segment_data = {'idx_start': seg_start, 'idx_end': seg_end, 'direction': seg_direction}
                            segments.append(segment_data)
                            break
                    # case when the segment hits a first valley that breaks the monotonous decreasing trend
                    else:
                        if df_PV_copy.iloc[i]['Low'] > df_PV_copy.iloc[i-2]['Low']:
                            seg_end = df_PV_copy.index[i-2]
                            segment_data = {'idx_start': seg_start, 'idx_end': seg_end, 'direction': seg_direction}
                            segments.append(segment_data)
                            break

                    # case when it is the end of the df_PV_copy
                    if df_PV_copy.index[i] == df_PV.index[-1]:
                        seg_end = df_PV_copy.index[-1]
                        segment_data = {'idx_start': seg_start, 'idx_end': seg_end, 'direction': seg_direction}
                        segments.append(segment_data)
                        break


        # update the df_PV_copy
        debug_logging(f'Processed segment: {seg_start} to {seg_end}')
        df_PV_copy = df_PV_copy.loc[seg_end:]

    # Create df_segments from the segments list
    df_segments = pd.DataFrame(segments)

    # Create a subset of df_PV that only contains the segment indexes
    index_to_keep_start = df_segments['idx_start'].tolist()
    index_to_keep_end = df_segments['idx_end'].tolist()
    index_to_keep = list(set(index_to_keep_start + index_to_keep_end))
    df_PV_segments = df_PV.loc[index_to_keep]
    df_PV_segments.sort_values(by=['Idx'], inplace=True)

    if df_PV_segments['pv_type'].iloc[-1] == df_PV_segments['pv_type'].iloc[-2]:
        # drop the last one - bandaid solution
        df_PV_segments = df_PV_segments.iloc[:-1]

    assert (df_PV_segments['pv_type'] != df_PV_segments['pv_type'].shift(1)).all()

    # refine the PVs
    df_PV_segments = refine_peaks_valleys(df_PV_segments, df_HL, debug_plot=False)

    # --- Visualization
    if debug_plot:
        debug_plot_segments(df_PV, df_HL, df_segments)
        debug_plot_peak_valley_factors(df_PV_segments, df_HL)

    return df_segments, df_PV_segments

def find_hubs(df_PV_segments, df_HL, df_OHLC, debug_plot=DEBUG_PLOT):

    # Simplify the df_PV_segments to only contain essential values
    df_PV_segments_orig = df_PV_segments.copy()
    df_PV_segments['factor_value'] = np.where(df_PV_segments['pv_type'] == 'Peak',
                                              df_PV_segments['High'],
                                              df_PV_segments['Low'])
    df_PV_segments['factor_value'] = df_PV_segments['factor_value'].astype(float)
    df_PV_segments.drop(['High', 'Low', 'is_peak', 'is_valley', 'pv_high', 'pv_low', 'direction'], axis=1, inplace=True)

    # Create a list to store the hubs
    list_hubs = []

    # Define a function to check if two line segments overlap
    def check_new_hub_formation(factor_0_type, factor_0, factor_1, factor_2, factor_3):
        hub_high = None
        hub_low = None
        is_new_hub = False
        if factor_0_type == 'Valley':   ## for a down hub
            is_new_hub = max(factor_0, factor_2) < min(factor_1, factor_3)
            if is_new_hub:
                hub_high = min(factor_1, factor_3)
                hub_low = max(factor_0, factor_2)
        elif factor_0_type == 'Peak':  ## for a up hub
            is_new_hub = min(factor_0, factor_2) > max(factor_1, factor_3)
            if is_new_hub:
                hub_high = min(factor_0, factor_2)
                hub_low = max(factor_1, factor_3)
        return is_new_hub, hub_high, hub_low

    def check_current_hub_belonging(cur_hub, factor_cur_type, factor_cur_value):
        if factor_cur_type == 'Valley':
            if factor_cur_value > cur_hub['high']:
                return False
            else:
                return True
        elif factor_cur_type == 'Peak':
            if factor_cur_value < cur_hub['low']:
                return False
            else:
                return True

    # initialize state variables
    last_included_factor_idx = -1
    in_hub = False

    for i in range(1, len(df_PV_segments) - 3):

        debug_logging('Processing factor: ' + str(i))
        if i < last_included_factor_idx:
            # Skip this iteration if it's part of an already processed hub
            continue

        # Get the current factor and the next three factors
        factor_0 = df_PV_segments.iloc[i]['factor_value']
        factor_1 = df_PV_segments.iloc[i + 1]['factor_value']
        factor_2 = df_PV_segments.iloc[i + 2]['factor_value']
        factor_3 = df_PV_segments.iloc[i + 3]['factor_value']
        factor_0_idx = df_PV_segments.index[i]
        factor_1_idx = df_PV_segments.index[i + 1]
        factor_2_idx = df_PV_segments.index[i + 2]
        factor_3_idx = df_PV_segments.index[i + 3]
        factor_0_type = df_PV_segments.iloc[i]['pv_type']

        # Case 1 - if not in a hub, then check if a new hub is formed
        if not in_hub:

            # Determine if the first three lines (current and next two factors) form a hub
            is_new_hub, hub_high, hub_low = check_new_hub_formation(factor_0_type, factor_0, factor_1, factor_2, factor_3)

            # if no new hub can be formed, continue to the next iteration
            if not is_new_hub:
                continue  # Skip to next iteration if not a hub
            # otherwise, define the new hub
            else:
                # If a new hub is formed, then set the state variables
                in_hub = True
                cur_hub = {'start_idx': factor_0_idx,
                           'end_idx': factor_3_idx,
                           'all_idx': [factor_0_idx, factor_1_idx, factor_2_idx, factor_3_idx],
                           'num_segments': 3,
                           'direction': 'down' if factor_0_type == 'Valley' else 'up',
                           'high': hub_high,
                           'low': hub_low}
                debug_logging(f'New hub formed from {factor_0_idx} to {factor_3_idx} with high {hub_high} and low {hub_low}')

                # continue to check if the next few factors belong to this hub
                for j in range(i + 4, len(df_PV_segments)):
                    factor_cur_value = df_PV_segments.iloc[j]['factor_value']
                    factor_cur_idx = df_PV_segments.index[j]
                    factor_cur_type = df_PV_segments.iloc[j]['pv_type']

                    # if the current factor belongs to the current hub, then update the hub
                    if check_current_hub_belonging(cur_hub, factor_cur_type, factor_cur_value):
                        cur_hub['end_idx'] = factor_cur_idx
                        cur_hub['num_segments'] += 1
                        cur_hub['all_idx'].append(factor_cur_idx)
                        debug_logging(f'Extended hub end to {factor_cur_idx}')
                    else:
                        # hub breaks
                        in_hub = False
                        list_hubs.append(cur_hub)
                        last_included_factor_idx = j - 1
                        debug_logging(f'Hub breaks at {factor_cur_idx}')
                        break

                    # case if it is the last factor
                    if j == len(df_PV_segments) - 1:
                        in_hub = False
                        list_hubs.append(cur_hub)
                        last_included_factor_idx = j
                        debug_logging(f'Hub breaks at {factor_cur_idx}')
                        break

    print(list_hubs)
    df_hubs = pd.DataFrame(list_hubs)

    # --- Visualization
    if debug_plot:
        # debug_plot_hubs_using_HL(df_PV_segments, df_HL, df_hubs)
        # debug_plot_hubs_using_OHLC(df_PV_segments, df_OHLC, df_hubs)
        fig_hubs = debug_plot_hubs(df_PV_segments, df_HL, df_OHLC, df_hubs)

    if debug_plot:
        return df_hubs, fig_hubs
    else:
        return df_hubs

def pattern_setup_trending_hubs_pull_back(df_hubs, df_OHLC, debug_plot=DEBUG_PLOT):
    """ This function identifies the pullback trading setup for trending hubs"""

    # Extract the last two hubs
    if len(df_hubs) < 2:
        return None
    else:
        hub_cur = df_hubs.iloc[-1]
        hub_prev = df_hubs.iloc[-2]

    # current price
    cur_price = df_OHLC.iloc[-1]['Close']

    # --- Check the relation between the two hubs
    # Case 1 - the last hub is higher than the previous hub (up trend)
    msg = 'No valid setup'
    if hub_cur['low'] > hub_prev['high']:
        if cur_price < hub_cur['low']:
            msg = 'Pullback long setup'
            return 1, msg  # pullback long setup
        else:
            return 0, msg  # no valid setup
    # Case 2 - the last hub is lower than the previous one (down trend
    elif hub_cur['high'] < hub_prev['low']:
        if cur_price > hub_cur['high']:
            msg = 'Pullback short setup'
            return -1, msg  # pullback short setup
        else:
            return 0, msg  # no valid setup

def pattern_setup_RSI_extreme(df_OHLC,
                              thres_overbought=65,
                              thres_oversold=35,
                              duration_overbought=6,
                              duration_oversold=6,
                              debug_plot=DEBUG_PLOT):
    import talib

    # Ensure df_OHLC is a standalone DataFrame
    df_OHLC = df_OHLC.copy()

    # Calculate RSI
    df_OHLC.loc[:, 'RSI'] = talib.RSI(df_OHLC['Close'], timeperiod=14)

    # label the RSI values that are overbought or oversold
    df_OHLC.loc[:, 'RSI_overbought'] = 0
    df_OHLC.loc[:, 'RSI_oversold'] = 0
    df_OHLC.loc[df_OHLC['RSI'] > thres_overbought, 'RSI_overbought'] = 1
    df_OHLC.loc[df_OHLC['RSI'] < thres_oversold, 'RSI_oversold'] = 1

    # Propagate the RSI values
    def propagate_values(df, len_prop):
        df.columns = ['value']
        i = 0
        while i < len(df):
            if df.iloc[i]['value'] != 0:
                value = df.iloc[i]['value']
                j = 1
                while j < len_prop + 1 and i + j < len(df) and df.iloc[i + j]['value'] == 0:
                    df.at[df.index[i + j], 'value'] = value
                    j += 1
                i += j
            else:
                i += 1
        return df['value']

    df_OHLC.loc[:, 'RSI_overbought_propagated'] = propagate_values(df_OHLC[['RSI_overbought']], duration_overbought)
    df_OHLC.loc[:, 'RSI_oversold_propagated'] = propagate_values(df_OHLC[['RSI_oversold']], duration_oversold)

    # --- Check if RSI is at an extreme
    if df_OHLC.iloc[-1]['RSI_oversold_propagated'] == 1:
        msg = 'RSI oversold long setup'
        return 1, msg  # long setup
    elif df_OHLC.iloc[-1]['RSI_overbought_propagated'] == 1:
        msg = 'RSI overbought long setup'
        return -1, msg  # short setup
    else:
        msg = 'No valid RSI overbought/sold setup'
        return 0, msg  # no valid setup


def main(df_OHLC_mid,
         num_candles=500,
         debug_plot=False,
         debug_print=False,
         use_high_low=False):

    """ This mid-timeframe strategy is based on the Chan Theory and contains the following steps:
    Step 1. Process the OHLC data with containing patterns. (包含关系)
    Step 2. Find peaks and valleys (fractal tops and bottoms) (顶底分型以及线段)
    Step 3. Find central zones (hubs) (寻找中枢）
    Step 4. Identify the trading setup (根据中枢分布决定交易形态）

    """

    # slice the DataFrame to the last 'num_candles' candles
    df_OHLC_mid = df_OHLC_mid.iloc[-num_candles:]

    # Step 1 - Apply containing patterns
    df_HL = apply_containing_pattern(df_OHLC_mid, use_high_low=use_high_low, debug_plot=debug_plot)

    # Step 2 - Extract line/segment peak/valleys
    df_PV_raw = find_raw_peaks_valleys(df_HL, debug_plot=debug_plot)
    _, df_PV_segments = find_segments(df_PV_raw, df_HL, debug_plot=debug_plot)

    # Step 3 - Identify hubs
    df_hubs, fig_hubs = find_hubs(df_PV_segments, df_HL, df_OHLC_mid, debug_plot=True)

    # Step 4 - Identify trading setup
    setup_decision_1, msg_1 = pattern_setup_trending_hubs_pull_back(df_hubs, df_OHLC_mid, debug_plot=True)
    debug_logging(f'Setup decision 1: {setup_decision_1}, {msg_1}')

    setup_decision_2, msg_2 = pattern_setup_RSI_extreme(df_OHLC_mid, debug_plot=debug_plot)
    debug_logging(f'Setup decision 2: {setup_decision_2}, {msg_2}')

    # Final decision
    final_decision = setup_decision_1 * setup_decision_2
    debug_logging(f'Final setup decision: {final_decision}')

    return final_decision, fig_hubs

