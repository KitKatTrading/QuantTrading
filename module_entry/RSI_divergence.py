import plotly
import os
import datetime
import numpy as np
import pandas as pd
import talib
from scipy.signal import argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def fun_findCrossing(diff: object, dea: object, check_under_thres: object, thres: object) -> object:
    """ DIFF = signal that will cross its own MA, for MACD this is the EMA12-EMA26
    :rtype: object
    """
    """ DEA  = this is the moving average of DIFF, based on user's input, to be crossed """
    """ checkUnderZero = if 1, only show gold cross when underwater and death cross when above water """
    crossing_no_zero_check = (diff - dea > 0).squeeze().astype(int).diff(1)

    if check_under_thres == 1:
        crossing_under_zero = (dea < thres).squeeze().astype(int)
        crossing_check_flag = abs(crossing_no_zero_check + crossing_under_zero - 0.5) > 1
        crossing_zero_check = crossing_check_flag * crossing_no_zero_check
        crossing_zero_check = crossing_zero_check.dropna().astype(int)
        return crossing_zero_check
    else:
        return crossing_no_zero_check

def find_peak_valley(data: np.array, order):
    '''
    Finds consecutive higher highs in price pattern.
    Must not be exceeded within the number of periods indicated by the width
    parameter for the value to be confirmed.
    K determines how many consecutive highs need to be higher.
    '''
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    return high_idx, highs, low_idx, lows

def find_divergence(
        data_temp,  # dataframe - with OHLC and index time
        indicator_name,  # string - name of the indicator, e.g., “MACD"
        thres_pv_local_extreme=5,  # int - number of neighbours to confirm a peak or a valley
        thres_bull_divergence_deflection=1,
        thres_bear_divergence_deflection=1,
        num_candles_alert=3,  # number of candles needed to confirm the divergence on the most recent pv
        num_candles=300,
        plot_results=1,
        save_results=1,
        soft_peak_comparison_ratio=0.1
):

    # set date format
    date_format = "%Y-%m-%d %H:%M:%S+00:00"
    date_format_filename = "%Y_%m_%d_%H_%M_%S"

    # start testing
    flag_bull_divergence = 0
    flag_bear_divergence = 0
    data_temp['flag_bull_divergence'] = [0] * len(data_temp.index)
    data_temp['flag_bear_divergence'] = [0] * len(data_temp.index)
    data_temp.loc[:, 'flag_bull_divergence'] = 0
    data_temp.loc[:, 'flag_bear_divergence'] = 0

    # print(date_test_end)
    for idx in range(num_candles, len(data_temp.index), 1):

        data_test = data_temp.iloc[idx-num_candles:idx+1]

        flag_bull_divergence_old = flag_bull_divergence
        flag_bear_divergence_old = flag_bear_divergence
        flag_bull_divergence = 0
        flag_bear_divergence = 0
        indicator_highs_idx_LH_aligned = np.empty(0)
        indicator_lows_idx_HL_aligned = np.empty(0)

        ### get the closing price and peak values
        price_close = data_test['Close'].values
        indicator = data_test[indicator_name].values
        dates = data_test.index

        # Subplot #1 - Price highs and lows
        indicator_highs_idx, indicator_highs, indicator_lows_idx, indicator_lows = find_peak_valley(indicator, thres_pv_local_extreme)

        # Secondary signal (indicator) - determine lower highs and higher lows for last N extremes
        num_indicator_LH = 0
        for idx_temp in range(len(indicator_highs) - 1, -1, -1):
            if indicator_highs[idx_temp] > indicator_highs[idx_temp - 1] and idx_temp - 1 >= 0:
                num_indicator_LH = num_indicator_LH + 1
                indicator_highs_LH = indicator_highs[idx_temp:]
                indicator_highs_idx_LH = indicator_highs_idx[idx_temp:]
                break
        if num_indicator_LH == 0:
            indicator_highs_LH = np.empty(0)
            indicator_highs_idx_LH = np.empty(0)

        num_indicator_HL = 0
        for idx_temp in range(len(indicator_lows) - 1, -1, -1):
            if indicator_lows[idx_temp] < indicator_lows[idx_temp - 1] and idx - 1 >= 0:
                num_indicator_HL = num_indicator_HL + 1
                indicator_lows_HL = indicator_lows[idx_temp:]
                indicator_lows_idx_HL = indicator_lows_idx[idx_temp:]
                break
        if num_indicator_HL == 0:
            indicator_lows_HL = np.empty(0)
            indicator_lows_idx_HL = np.empty(0)

        # check for bullish divergence - price LL while indicator HL
        if num_indicator_HL != 0:
            price_close_temp = price_close[indicator_lows_idx_HL]
            # check for LL in price
            num_aligned = 0
            for idx_temp in range(len(indicator_lows_idx_HL) - 1, -1, -1):
                if (1-soft_peak_comparison_ratio) * price_close_temp[idx_temp] < price_close_temp[idx_temp - 1]:
                    num_aligned = num_aligned + 1
                else:
                    break
            if num_aligned >= 1:
                indicator_lows_idx_HL_aligned = indicator_lows_idx_HL[idx_temp:]
            if num_aligned >= thres_bull_divergence_deflection and indicator_lows_idx_HL_aligned[-1] == num_candles - num_candles_alert - 1:
                flag_bull_divergence = 1
                data_temp.at[dates[-1], 'flag_bull_divergence'] = 1
            else:
                flag_bull_divergence = 0
                data_temp.at[dates[-1], 'flag_bull_divergence'] = 0

        # check for bearish divergence - price HH while indicator LH
        if num_indicator_LH != 0:
            price_close_temp = price_close[indicator_highs_idx_LH]
            # check for HH in price
            num_aligned = 0
            for idx_temp in range(len(indicator_highs_idx_LH) - 1, -1, -1):
                if price_close_temp[idx_temp] > (1-soft_peak_comparison_ratio) * price_close_temp[idx_temp - 1]:
                    num_aligned = num_aligned + 1
                else:
                    break
            if num_aligned >= 1:
                indicator_highs_idx_LH_aligned = indicator_highs_idx_LH[idx_temp:]
            if num_aligned >= thres_bear_divergence_deflection and indicator_highs_idx_LH[-1] == num_candles - num_candles_alert - 1:
                flag_bear_divergence = 1
                data_temp.at[dates[-1], 'flag_bear_divergence'] = 1
            else:
                flag_bear_divergence = 0
                data_temp.at[dates[-1], 'flag_bear_divergence'] = 0

    if plot_results == 1 and ((flag_bull_divergence == 1) or (flag_bear_divergence==1)):

        # initialize figure
        fig = make_subplots(rows=2, cols=1)
        fig.update_layout(xaxis_rangeslider_visible=False)

        # plot price - master signal
        fig.add_trace(
            go.Candlestick(x=data_test.index,
                           open=data_test['Open'],
                           close=data_test['Close'],
                           low=data_test['Low'],
                           high=data_test['High'],
                           showlegend=False),
            row=1, col=1)

        # plot indicator - MACD in this case

        # if MACD or MACD histogram divergence
        if indicator_name == 'MACD' or indicator_name == 'MACD_hist':
            colors = ['green' if val >= 0 else 'red' for val in data_test['MACD_hist']]
            fig.add_trace(
                go.Bar(x=data_test.index,
                       y=data_test['MACD_hist'],
                       marker_color=colors,
                       name='MACD_hist',
                       legendgroup='3'),
                row=2, col=1)
            fig.add_trace(
                go.Scatter(x=data_test.index,
                           y=data_test['MACD'],
                           line=dict(color='blue', width=1),
                           mode="lines",
                           name='MACD',
                           legendgroup='3'),
                row=2, col=1)
            fig.add_trace(
                go.Scatter(x=data_test.index,
                           y=data_test['MACD_signal'],
                           line=dict(color='orange', width=1),
                           mode="lines",
                           name='MACD_signal',
                           legendgroup='3'),
                row=2, col=1)
        if indicator_name == 'RSI':
            fig.add_trace(
                go.Scatter(x=data_test.index,
                           y=data_test['RSI'],
                           line=dict(color='black', width=1),
                           mode="lines",
                           name='RSI',
                           legendgroup='3'),
                row=2, col=1)

        if len(indicator_lows_idx_HL_aligned) != 0:
            # lower lows in price
            fig.add_trace(
                go.Scatter(x=dates[indicator_lows_idx_HL_aligned],
                           y=price_close[indicator_lows_idx_HL_aligned],
                           name="price_LL",
                           mode="lines",
                           legendgroup='2',
                           line=dict(color="green")),
                row=1, col=1)

            # higher lows in indicator
            fig.add_trace(
                go.Scatter(x=dates[indicator_lows_idx_HL_aligned],
                           y=data_test[indicator_name][indicator_lows_idx_HL_aligned],
                           name="indicator_HL",
                           mode="lines",
                           legendgroup='2',
                           line=dict(color="green", width=2)),
                row=2, col=1)

        if len(indicator_highs_idx_LH_aligned) != 0:
            # higher highs in price
            fig.add_trace(
                go.Scatter(x=dates[indicator_highs_idx_LH_aligned],
                           y=price_close[indicator_highs_idx_LH_aligned],
                           name="price_HH",
                           mode="lines",
                           legendgroup='3',
                           line=dict(color="red")),
                row=1, col=1)

            # lower highs in indicator
            fig.add_trace(
                go.Scatter(x=dates[indicator_highs_idx_LH_aligned],
                           y=data_test[indicator_name][indicator_highs_idx_LH_aligned],
                           name="indicator_LH",
                           mode="lines",
                           legendgroup='3',
                           line=dict(color="red", width=2)),
                row=2, col=1)

        # export plot
        fig.update_layout(legend=dict(orientation="h", xanchor="center", yanchor="top", y=-0.1, x=0.5))
        fig.update_layout(showlegend=False)
        # fig.show()
        if save_results == 1:
            date_test_end_str_filename = datetime.datetime.strptime(dates[-1], date_format).strftime(date_format_filename)
            # fig.write_image(os.path.join('Plot', 'Divergence', 'Bull_' + str(flag_bull_divergence) + '_Bear_' + str(flag_bear_divergence) + '_' + date_test_end_str_filename + '.png'), scale=1,width=1920, height=1080)
            fig.write_image(os.path.join('Bull_' + str(flag_bull_divergence) + '_Bear_' + str(flag_bear_divergence) + '_' + date_test_end_str_filename + '.png'), scale=1,width=1920, height=1080)

        return fig, data_temp['flag_bull_divergence'], data_temp['flag_bear_divergence']
    else:
        return [], data_temp['flag_bull_divergence'], data_temp['flag_bear_divergence']

def compute_indicator(pd_data, plot_results=0, save_results=0, num_candles=200, thres_pv_local_extreme=5, do_propagation=True):
    pd_data = pd_data.copy()
    pd_data['RSI'] = talib.RSI(pd_data['Close'], 14)
    pd_data = pd_data.dropna()
    fig, pd_data_long, pd_data_short = find_divergence(
        pd_data, 'RSI',
        thres_pv_local_extreme=thres_pv_local_extreme,
        thres_bull_divergence_deflection=1,
        thres_bear_divergence_deflection=1,
        num_candles=num_candles,
        plot_results=plot_results,
        save_results=save_results)

    pd_data_RSI = pd_data_long - pd_data_short
    pd_data_RSI = pd_data_RSI.to_frame()
    pd_data_RSI.columns = ['value']

    if do_propagation:
       propagate_values(pd_data_RSI, 2)

    my_series = pd_data_RSI.squeeze()

    return my_series, fig

def propagate_values(df, len_prop):
    i = 0
    while i < len(df):
        if df.iloc[i]['value'] != 0:
            value = df.iloc[i]['value']
            j = 1
            while j < len_prop+1 and i + j < len(df) and df.iloc[i + j]['value'] == 0:
                df.at[df.index[i + j], 'value'] = value
                j += 1
            i += j
        else:
            i += 1

def main(df_OHLC, num_candles=200, run_mode='live'):

    if run_mode == 'live':

        # slice the DataFrame to the last 'num_candles' candles
        df_OHLC = df_OHLC.iloc[-num_candles:]

        # compute the indicator
        my_series, fig = compute_indicator(df_OHLC, plot_results=1, save_results=1, num_candles=num_candles,
                                      thres_pv_local_extreme=5)

        return my_series.iloc[-1], fig

    elif run_mode == 'backtest':

        df_decision = pd.DataFrame(0, index=df_OHLC.index, columns=['value'])
        my_series, fig = compute_indicator(df_OHLC, plot_results=1, save_results=1, num_candles=num_candles,
                                      thres_pv_local_extreme=5, do_propagation=False)

        df_decision['value'] = my_series
        df_decision.dropna(inplace=True)

        return df_decision, fig


