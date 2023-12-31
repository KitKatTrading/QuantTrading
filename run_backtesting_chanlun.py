from Objects.backtesting import *
from Objects.pnl_analysis import *

# read the config file "config_local_path.py"
import config_local_path
dir_data = config_local_path.gvars['dir_module_data_crypto_binance']
dir_backtesting = config_local_path.gvars['dir_module_backtest']

# get the current datetime
datetime_now_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
datetime_now = datetime.utcnow()
datetime_now_rounded = datetime_now.replace(minute=0, second=0, microsecond=0)
datetime_now_rounded = datetime_now_rounded.strftime("%Y-%m-%d %H:%M:%S+00:00")

if __name__ == '__main__':

    # Define the strategy - high timeframe for bias
    name_strategy_high_timeframe = 'SMA_price_10_20_trend'
    timeframe_high = '1d'

    # Define the strategy - mid timeframe for pattern
    name_strategy_mid_timeframe = 'chanlun_poway'
    timeframe_mid = '1h'

    # Define the strategy - low timeframe for entry
    name_strategy_low_timeframe = 'RSI_extreme_cross'
    timeframe_low = '1h'

    # Define the backtesting start and end dates
    bt_start_date = "2021-01-01 00:00:00+00:00"
    bt_start_date = "2023-01-01 00:00:00+00:00"
    bt_end_date = datetime_now_rounded

    # Define the name of the backtesting
    name_backtesting = os.path.join(f"{datetime_now_str}_"
                                    f"{name_strategy_high_timeframe}_{timeframe_high}_"
                                    f"{name_strategy_mid_timeframe}_{timeframe_mid}_"
                                    f"{name_strategy_low_timeframe}_{timeframe_low}")

    # Define the list of names and symbols
    names_symbol = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT',
                    'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT', 'NEARUSDT', 'RUNEUSDT', 'OPUSDT', 'INJUSDT',
                    'LDOUSDT', 'EGLDUSDT', 'THETAUSDT', 'FTMUSDT', 'SANDUSDT', 'GALAUSDT', 'XTZUSDT', 'EOSUSDT',
                    'LTCUSDT', 'BCHUSDT', 'ZECUSDT', 'SEIUSDT', 'FILUSDT', 'DOTUSDT', 'LINKUSDT', 'AAVEUSDT',
                    'OCEANUSDT', 'AGLDUSDT', 'TRBUSDT', 'ALICEUSDT', 'XMRUSDT', 'XLMUSDT', 'DYDXUSDT', 'ICPUSDT',
                    'VETUSDT', 'SUSHIUSDT', 'KSMUSDT', 'GRTUSDT', '1INCHUSDT', 'ZENUSDT', 'YFIUSDT', 'BATUSDT',
                    'SNXUSDT', 'MKRUSDT', 'COMPUSDT', 'ENJUSDT', 'RENUSDT', 'CRVUSDT', 'MANAUSDT', 'MASKUSDT',
                    'CELRUSDT', 'OGNUSDT', 'REEFUSDT', 'DENTUSDT', 'RVNUSDT', 'DODOUSDT', 'HNTUSDT', 'TOMOUSDT',
                    'LITUSDT', 'COTIUSDT', 'AUDIOUSDT', 'AKROUSDT', 'CVCUSDT', 'STORJUSDT', 'HOTUSDT', 'NKNUSDT',
                    'WAVESUSDT', 'KAVAUSDT', 'ALGOUSDT', 'NEOUSDT', 'QTUMUSDT']
    # names_symbol = ['EGLDUSDT', 'AVAXUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'INJUSDT', 'OPUSDT']
    # names_symbol = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'RUNEUSDT', 'OPUSDT', 'AVAXUSDT']
    names_symbol = ['BTCUSDT', 'OPUDST', 'NEARUSDT', 'SOLUSDT', 'AVAXUSDT']

    # initialize master dataframe for all symbols
    df_master = pd.DataFrame(columns=['name_symbol', 'num_trades', 'num_wins', 'num_losses', 'num_breakeven',
                                      'avg_rrr_win', 'avg_rrr_lose', 'avg_rrr_wl', 'avg_rrr_ratio',
                                      'win_rate', 'win_rate_be', 'avg_duration'])

    for idx, name_symbol in enumerate(names_symbol):
        print(f"Processing {name_symbol}:")

        try:
            # Define the backtesting object
            backtesting = Backtesting(name_symbol=name_symbol, data_source='binance', name_strategy=name_backtesting,
                                      save_plot=True, save_csv=True,
                                      timeframe_high=timeframe_high,
                                      timeframe_mid=timeframe_mid,
                                      timeframe_low=timeframe_low,
                                      function_high_timeframe=name_strategy_high_timeframe,
                                      function_mid_timeframe=name_strategy_mid_timeframe,
                                      function_low_timeframe=name_strategy_low_timeframe,
                                      dir_data=dir_data,
                                      dir_backtesting=dir_backtesting,
                                      bt_start_date=bt_start_date,
                                      bt_end_date=bt_end_date)
                                      # bt_end_date='2023-07-01 00:00:00+00:00')

            # Identify the entries
            print("-- identifying entries...")
            backtesting.find_entries_vectorize_high_low(manual_review_each_trade=False)

            # Execute the trades
            print("-- executing trades...")
            df_trade_log = backtesting.execute_trades(save_csv=True)
            # df_trade_log has these columns for each trade (one row per trade):
            # 'entry_number', 'direction', 'datetime_entry', 'entry_idx', 'entry_price', 'initial_risk', 'datetime_exit',
            # 'exit_idx', 'exit_price', 'max_profit', 'pnl', 'pnl_max', 'rrr', 'rrr_max', 'duration', 'win_loss'

            # Add the trade log to the master dataframe
            # udpate the trade results dataframe
            num_wins = len(df_trade_log[df_trade_log['win_loss'] == 'W'])
            num_losses = len(df_trade_log[df_trade_log['win_loss'] == 'L'])
            avg_rrr_win = df_trade_log[df_trade_log['win_loss'] == 'W']['rrr'].mean()
            avg_rrr_lose = df_trade_log[df_trade_log['win_loss'] == 'L']['rrr'].mean()
            avg_rrr_wl = df_trade_log[df_trade_log['win_loss'] != 'B']['rrr'].mean()
            avg_rrr_ratio = abs(avg_rrr_win / avg_rrr_lose)
            win_rate = num_wins / (num_wins + num_losses)
            win_rate_be = 1 / (1 + avg_rrr_ratio)
            cur_symbol = {
                'name_symbol': name_symbol,
                'num_trades': len(df_trade_log),
                'num_wins': len(df_trade_log[df_trade_log['win_loss'] == 'W']),
                'num_losses': len(df_trade_log[df_trade_log['win_loss'] == 'L']),
                'num_breakeven': len(df_trade_log[df_trade_log['win_loss'] == 'E']),
                'avg_rrr_win': avg_rrr_win,
                'avg_rrr_lose': avg_rrr_lose,
                'avg_rrr_ratio': avg_rrr_ratio,
                'avg_rrr_wl': avg_rrr_wl,
                'win_rate': win_rate,
                'win_rate_be': win_rate_be,
                'avg_duration': df_trade_log['duration'].mean()
            }
            df_master = pd.concat([df_master, pd.DataFrame(cur_symbol, index=[idx])])

        except:
            print(f"Error processing {name_symbol}.")
    df_master.to_csv(os.path.join(dir_backtesting, name_backtesting, 'backtesting_results.csv'), index=False)

    # Run pnl analysis
    pnl_analysis = PNL(job_name=backtesting.backtesting_dir_strategy,
                       datetime_bt_start=backtesting.bt_start_date,
                       datetime_bt_end=backtesting.bt_end_date,
                       timeframe_low=backtesting.timeframe_low
                       )

    fig_pnl = pnl_analysis.visualize_pnl_curve()
    fig_pnl.show()
    fig_pnl.write_html('pnl_curve.html')

