from Objects.backtesting import *

if __name__ == '__main__':

    # Get the current datetime
    datetime_now = datetime.utcnow()
    datetime_now_rounded = datetime_now.replace(minute=0, second=0, microsecond=0)
    datetime_now_rounded = datetime_now_rounded.strftime("%Y-%m-%d %H:%M:%S+00:00")

    # define the list of names and symbols
    names_symbol = ['EGLDUSDT', 'AVAXUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'INJUSDT', 'OPUSDT']
    # names_symbol = ['EGLDUSDT', 'AVAXUSDT']
    # names_symbol = ['EGLDUSDT']

    for name_symbol in names_symbol:
        print(f"Processing {name_symbol}:")

        # Define the backtesting object
        backtesting = Backtesting(name_symbol=name_symbol, data_source='binance', name_strategy='chanlun',
                                  timeframe_high='1w', timeframe_mid='1h', timeframe_low='1h',
                                  function_high_timeframe='always_long',
                                  function_mid_timeframe='chanlun',
                                  function_low_timeframe='RSI_extreme_cross',
                                  bt_start_date='2021-01-01 00:00:00+00:00',
                                  # bt_end_date=datetime_now_rounded)
                                  bt_end_date='2023-12-01 00:00:00+00:00')

        # Identify the entries
        print("-- identifying entries...")
        df_entry_log = backtesting.find_entries_vectorize_high_low(manual_review_each_trade=False,
                                                                   save_plots=True,
                                                                   save_csv=True)

        # Execute the trades
        print("-- executing trades...")
        df_bt_result = backtesting.execute_trades(save_csv=True)


