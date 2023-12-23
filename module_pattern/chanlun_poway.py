from Objects.chanlun_hubs import *
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def main(df_OHLC_mid,
         name_symbol,
         time_frame,
         num_candles=500,
         # use_high_low=False,
         ):

    # convert DataFrame to bars object
    df_OHLC_mid = df_OHLC_mid[-num_candles:]
    bars = convert_df_to_bars(df_OHLC_mid, time_frame, name_symbol)

    # Call the Chan Analysis object (the signal calculation is inside the object)
    # function "pattern_setup_single_hub_poway"
    try:
        chanlun = CZSC(bars,
                       symbol=name_symbol,
                       freq=time_frame,
                       )
        # chanlun.chart = chanlun.to_echarts()
        chanlun.chart = chanlun.to_plotly()
        # chanlun.chart.show()
        return chanlun.decision, chanlun.chart

    except Exception as e:
        print(e)
        return 0, None



