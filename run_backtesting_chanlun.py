from Objects.backtesting import *

if __name__ == '__main__':

    # Define the backtesting object
    backtesting = Backtesting(name_symbol='AVAXUSDT', data_source='binance', name_strategy='chanlun',
                              timeframe_high='1w', timeframe_mid='1h', timeframe_low='1h',
                              function_high_timeframe='always_long',
                              function_mid_timeframe='chanlun',
                              function_low_timeframe='RSI_extreme_cross',
                              bt_start_date='2023-01-01 00:00:00+00:00',
                              bt_end_date='2023-12-13 00:00:00+00:00')

    # Run the backtesting
    trading_decision = backtesting.run_backtesting_vectorize_high_low()
    print(trading_decision)
    # debug_logging(trading_decision)
