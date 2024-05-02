from objects.chanlun_hubs import *
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def main(df_OHLC_mid,
         name_symbol,
         time_frame,
         num_candles=500,
         remove_fake_pins=True,
         ):

    # convert DataFrame to bars object
    df_OHLC_mid = df_OHLC_mid[-num_candles:]
    bars = convert_df_to_bars(df_OHLC_mid, time_frame, name_symbol)


    # main analysis


