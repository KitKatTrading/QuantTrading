# import os
# import pandas as pd
# from datetime import datetime, timedelta
# from utils.util_general import debug_logging
from objects.class_strategy import *
from objects.class_backtesting import *

# Adjust dir_data and add dir_utils to the Python path
current_script_dir = os.path.dirname(__file__)
dir_data = os.path.join(current_script_dir, '', '../module_data')
dir_data = os.path.normpath(dir_data)
dir_backtesting = os.path.join(current_script_dir, '', '../module_backtesting')



if __name__ == '__main__':

    # Define the backtesting object
    backtesting = Backtesting(name_symbol='AVAXUSDT', data_source='binance', name_strategy='chanlun',
                              timeframe_high='1w', timeframe_mid='12h', timeframe_low='1h',
                              function_high_timeframe='always_long',
                              function_mid_timeframe='chanlun',
                              function_low_timeframe='RSI_extreme_cross',
                              bt_start_date='2023-01-01 00:00:00+00:00',
                              bt_end_date='2023-12-09 00:00:00+00:00')

    # Run the backtesting
    trading_decision = backtesting.run_backtesting_vectorize_high_low()
    print(trading_decision)
    # debug_logging(trading_decision)
