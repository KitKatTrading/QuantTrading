import pandas as pd
import talib
import plotly

def main(pd_OHLC_mid, num_candles=200):

    """ This mid-timeframe strategy is based on the consolidation zone trading ideas. The strategy is based on the
    following steps:
    Step 1. Identify the highs and lows of the given OHLC data.
    Step 2. Draw connections of adjacent highs and lows, and identify conslidation and trend regions.
    Step 3. The specific trading opportunity is triggered then the price falls bcak to the edge of the zone.
    For example, if the high timeframe direction is Long, then we should Long when the price falls to the lower edge.
    """

    # slice the DataFrame to the last 'num_candles' candles
    pd_OHLC_mid = pd_OHLC_mid.iloc[-num_candles:]

    # find all peaks for highs and lows, including both the open and close prices
    pd_OHLC_mid['highs'] = pd_OHLC_mid[['Open', 'Close']].max(axis=1).rolling(5).max()
    pd_OHLC_mid['lows'] = pd_OHLC_mid[['Open', 'Close']].min(axis=1).rolling(5).min()

    # clean up the highs and lows such that there is always a peak between two valleys, and vice versa
    pd_OHLC_mid['highs'] = pd_OHLC_mid['highs'].fillna(method='ffill')
    pd_OHLC_mid['lows'] = pd_OHLC_mid['lows'].fillna(method='ffill')

    # plot the OHLC and visualize the highs and lows using plotly
    fig = plotly.graph_objs.Figure(data=[plotly.graph_objs.Candlestick(x=pd_OHLC_mid.index,
                                                                       open=pd_OHLC_mid['Open'],
                                                                       high=pd_OHLC_mid['High'],
                                                                       low=pd_OHLC_mid['Low'],
                                                                       close=pd_OHLC_mid['Close'])])
    fig.add_trace(plotly.graph_objs.Scatter(x=pd_OHLC_mid.index, y=pd_OHLC_mid['highs'], mode='markers'))
    fig.add_trace(plotly.graph_objs.Scatter(x=pd_OHLC_mid.index, y=pd_OHLC_mid['lows'], mode='markers'))
    fig.show()

    #




    # # identify the consolidations zones, which are featured by overalpping highs and lows
    # num_zones = 0
    # for i in range(len(pd_OHLC_mid)):
    #     if pd_OHLC_mid['highs'][i] > pd_OHLC_mid['lows'][i]:
    #         pd_OHLC_mid.loc[i, 'zone'] = num_zones
    #     else:
    #         num_zones += 1
    #         pd_OHLC_mid.loc[i, 'zone'] = num_zones

    return 0