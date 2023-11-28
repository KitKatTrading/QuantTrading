import plotly.graph_objs as go
from scipy.signal import find_peaks
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

DEBUG_PLOT = True

def generate_raw_peaks_and_valleys(df_OHLC_mid,
                                   use_high_low=False,
                                   scipy_pv_search_radius=5):

    # --- find peaks and valleys and organize them into a DataFrame
    # convert the 'Open' and 'Close' to a single series representing potential highs and lows
    if use_high_low:
        combined_prices_peak = df_OHLC_mid[['High', 'Low']].max(axis=1)
        combined_prices_valley = df_OHLC_mid[['High', 'Low']].min(axis=1)
    else:
        combined_prices_peak = df_OHLC_mid[['Open', 'Close']].max(axis=1)
        combined_prices_valley = df_OHLC_mid[['Open', 'Close']].min(axis=1)

    # Plot combined_prices_peak and combined_prices_valley with OHLC
    fig_peak_valley = go.Figure()

    # Add OHLC data
    fig_peak_valley.add_trace(go.Candlestick(x=df_OHLC_mid.index,
                                             open=df_OHLC_mid['Open'],
                                             high=df_OHLC_mid['High'],
                                             low=df_OHLC_mid['Low'],
                                             close=df_OHLC_mid['Close'],
                                             name='OHLC'))

    # Add Combined Peaks
    fig_peak_valley.add_trace(go.Scatter(x=df_OHLC_mid.index, y=combined_prices_peak,
                                         mode='lines', name='Combined Peaks',
                                         line=dict(color='red')))

    # Add Combined Valleys
    fig_peak_valley.add_trace(go.Scatter(x=df_OHLC_mid.index, y=combined_prices_valley,
                                         mode='lines', name='Combined Valleys',
                                         line=dict(color='blue')))

    fig_peak_valley.update_layout(title='Combined Prices Peak and Valleys with OHLC',
                                  xaxis_rangeslider_visible=False, showlegend=True,
                                  xaxis_title='Date', yaxis_title='Price')

    fig_peak_valley.show()

    # find peaks (highs) and valleys (lows)
    id_peaks, _ = find_peaks(combined_prices_peak, distance=scipy_pv_search_radius)
    id_valleys, _ = find_peaks(-combined_prices_valley, distance=scipy_pv_search_radius)

    # extract peak and valley values
    df_peaks = combined_prices_peak.iloc[id_peaks]
    df_valleys = combined_prices_valley.iloc[id_valleys]

    # create DataFrames from df_peaks and df_valleys and add a column to indicate the original index
    df_peaks = pd.DataFrame({'Date': df_peaks.index, 'Price': df_peaks.values, 'Type': 'Peak', 'orig_index': id_peaks})
    df_valleys = pd.DataFrame({'Date': df_valleys.index, 'Price': df_valleys.values, 'Type': 'Valley', 'orig_index': id_valleys})

    # add the index
    df_peaks.set_index('Date', inplace=True)
    df_valleys.set_index('Date', inplace=True)

    # combine peaks and valleys into one DataFrame
    df_pv_combined = pd.concat([df_peaks, df_valleys])

    # sort by index (which is the date in this case)
    df_pv_combined.sort_values(by='Date', inplace=True)

    return df_pv_combined


def process_raw_peaks_and_valleys(df_pv_combined,
                                  min_pv_dist=3,
                                  ):
    """ This function does two things:
    1. Combine consecutive peaks or valleys.
    2. Drop the peaks and valleys that are too close to each other < min_pv_dist.
    """

    df_pv_combined.to_csv('df_pv_combined.csv')

    while True:

        # Step 1 - Check for consecutive same types using vectorized methods
        df_pv_combined['Next_Type'] = df_pv_combined['Type'].shift(-1)
        has_consecutive_same_type = (df_pv_combined['Type'] == df_pv_combined['Next_Type']).any()

        # Step 2 - Check for peaks and valleys that are too close
        df_pv_combined['Next_Index'] = df_pv_combined['orig_index'].shift(-1)
        too_close = ((df_pv_combined['Next_Index'] - df_pv_combined['orig_index']) < min_pv_dist).any()
        too_close = False


        # Decide if we can stop the loop
        if not has_consecutive_same_type and not too_close:
            break  # Exit the loop if no more processing is needed

        # Process consecutive same types
        if has_consecutive_same_type:
            new_rows = []
            skip_next = False
            for i in range(len(df_pv_combined)):
                if skip_next:
                    skip_next = False
                    continue

                if i < len(df_pv_combined) - 1 and df_pv_combined.iloc[i]['Type'] == df_pv_combined.iloc[i + 1]['Type']:
                    # For peaks, keep the row with the higher price; for valleys, keep the row with the lower price
                    if (df_pv_combined.iloc[i]['Type'] == 'Peak' and df_pv_combined.iloc[i]['Price'] <
                        df_pv_combined.iloc[i + 1]['Price']) \
                            or (df_pv_combined.iloc[i]['Type'] == 'Valley' and df_pv_combined.iloc[i]['Price'] >
                                df_pv_combined.iloc[i + 1]['Price']):
                        new_rows.append(df_pv_combined.iloc[i + 1])
                    else:
                        new_rows.append(df_pv_combined.iloc[i])
                    skip_next = True
                else:
                    new_rows.append(df_pv_combined.iloc[i])

            df_pv_combined = pd.concat(new_rows, axis=1).transpose()



    # Clean up helper columns
    df_pv_combined.drop(['Next_Type', 'Next_Index'], axis=1, inplace=True)
    df_pv_processed = df_pv_combined

    return df_pv_processed

def apply_containing_pattern(df_OHLC_mid,
                             use_high_low=False,
                             flag_plot=DEBUG_PLOT,
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
            if flag_plot:
                fig = go.Figure(data=[go.Candlestick(x=df_HL.index,
                                                     open=df_HL['Low'],
                                                     high=df_HL['High'],
                                                     low=df_HL['Low'],
                                                     close=df_HL['High'],
                                                     name='HL')])
                fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
                fig.show()

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
        if flag_plot:
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

def main(df_OHLC_mid, num_candles=200,
         scipy_pv_search_radius=5,
         min_pv_dist=4,
         use_high_low=False):

    """ This mid-timeframe strategy is based on the Chan Theory and contains the following steps:
    Step 1. Process the OHLC data with containing patterns. (包含关系)
    Step 2.
    Step 3.

    """

    # slice the DataFrame to the last 'num_candles' candles
    df_OHLC_mid = df_OHLC_mid.iloc[-num_candles:]

    # Step 1 - Apply containing patterns
    df_HL = apply_containing_pattern(df_OHLC_mid, use_high_low=use_high_low)

    # Step 2 - Find peaks and valleys (fractal tops and bottoms)






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