from Objects.backtesting import *

datetime_now_str = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

name_backtesting = 'chanlun' + datetime_now_str
name_strategy_high_timeframe = 'always_long'
name_strategy_mid_timeframe = 'chanlun'
name_strategy_low_timeframe = 'RSI_extreme_cross'


if __name__ == '__main__':

    # Get the current datetime
    datetime_now = datetime.utcnow()
    datetime_now_rounded = datetime_now.replace(minute=0, second=0, microsecond=0)
    datetime_now_rounded = datetime_now_rounded.strftime("%Y-%m-%d %H:%M:%S+00:00")

    # define the list of names and symbols
    names_symbol = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT',
                    'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT', 'NEARUSDT', 'RUNEUSDT', 'OPUSDT', 'INJUSDT',
                    'LDOUSDT', 'EGLDUSDT', 'THETAUSDT', 'FTMUSDT', 'SANDUSDT', 'GALAUSDT', 'XTZUSDT', 'EOSUSDT',
                    'LTCUSDT', 'BCHUSDT', 'ZECUSDT', 'SEIUSDT', 'FILUSDT', 'DOTUSDT', 'LINKUSDT', 'AAVEUSDT',
                    'OCEANUSDT', 'AGLDUSDT', 'TRBUSDT', 'ALICEUSDT', 'XMRUSDT', 'XLMUSDT',
                    'VETUSDT', 'SUSHIUSDT', 'KSMUSDT', 'GRTUSDT', '1INCHUSDT', 'ZENUSDT', 'YFIUSDT', 'BATUSDT',
                    'SNXUSDT', 'MKRUSDT', 'COMPUSDT', 'ENJUSDT', 'RENUSDT', 'CRVUSDT', 'MANAUSDT', 'MASKUSDT',
                    'CELRUSDT', 'OGNUSDT', 'REEFUSDT', 'DENTUSDT', 'RVNUSDT', 'DODOUSDT', 'HNTUSDT', 'TOMOUSDT',
                    'LITUSDT', 'COTIUSDT', 'AUDIOUSDT', 'AKROUSDT', 'CVCUSDT', 'STORJUSDT', 'HOTUSDT', 'NKNUSDT',
                    'WAVESUSDT', 'KAVAUSDT', 'ALGOUSDT', 'NEOUSDT', 'QTUMUSDT']
    # names_symbol = ['EGLDUSDT', 'AVAXUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'INJUSDT', 'OPUSDT']
    # names_symbol = ['EGLDUSDT']

    # initialize master dataframe for all symbols
    df_master = pd.DataFrame(columns=['name_symbol', 'num_trades', 'num_wins', 'num_losses', 'num_breakeven',
                                      'avg_rrr_win', 'avg_rrr_lose', 'avg_rrr_all', 'avg_duration'])

    for idx, name_symbol in enumerate(names_symbol):
        print(f"Processing {name_symbol}:")

        # Define the backtesting object
        backtesting = Backtesting(name_symbol=name_symbol, data_source='binance', name_strategy=name_backtesting,
                                  timeframe_high='1w', timeframe_mid='1h', timeframe_low='1h',
                                  function_high_timeframe=name_strategy_high_timeframe,
                                  function_mid_timeframe=name_strategy_mid_timeframe,
                                  function_low_timeframe=name_strategy_low_timeframe,
                                  bt_start_date='2021-01-01 00:00:00+00:00',
                                  bt_end_date=datetime_now_rounded)
                                  # bt_end_date='2023-07-01 00:00:00+00:00')

        # Identify the entries
        print("-- identifying entries...")
        backtesting.find_entries_vectorize_high_low(manual_review_each_trade=False,
                                                    save_plots=True,
                                                    save_csv=False)

        # Execute the trades
        print("-- executing trades...")
        df_trade_log = backtesting.execute_trades(save_csv=True)
        # df_trade_log has these columns for each trade (one row per trade):
        # 'entry_number', 'direction', 'datetime_entry', 'entry_idx', 'entry_price', 'initial_risk', 'datetime_exit',
        # 'exit_idx', 'exit_price', 'max_profit', 'pnl', 'pnl_max', 'rrr', 'rrr_max', 'duration', 'win_loss'

        # Add the trade log to the master dataframe
        # udpate the trade results dataframe
        avg_rrr_win = df_trade_log[df_trade_log['win_loss'] == 'W']['rrr'].mean()
        avg_rrr_lose = df_trade_log[df_trade_log['win_loss'] == 'L']['rrr'].mean()
        avg_rrr_all = abs(avg_rrr_win / avg_rrr_lose)
        cur_symbol = {
            'name_symbol': name_symbol,
            'num_trades': len(df_trade_log),
            'num_wins': len(df_trade_log[df_trade_log['win_loss'] == 'W']),
            'num_losses': len(df_trade_log[df_trade_log['win_loss'] == 'L']),
            'num_breakeven': len(df_trade_log[df_trade_log['win_loss'] == 'B']),
            'avg_rrr_win': avg_rrr_win,
            'avg_rrr_lose': avg_rrr_lose,
            'avg_rrr_all': avg_rrr_all,
            'avg_duration': df_trade_log['duration'].mean()
        }
        df_master = pd.concat([df_master, pd.DataFrame(cur_symbol, index=[idx])])

    df_master.to_csv(os.path.join('module_backtesting', name_backtesting, 'backtesting_results.csv'), index=False)
