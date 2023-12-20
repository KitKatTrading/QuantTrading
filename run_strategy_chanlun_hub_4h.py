from Objects.strategy import Strategy

strategy_chanlun = Strategy(name_symbol='OPUSDT', data_source='binance', name_strategy='chanlun_12h',
                            timeframe_high='1d', timeframe_mid='4h', timeframe_low='1h',
                            function_high_timeframe='always_long',
                            function_mid_timeframe='chanlun',
                            function_low_timeframe='RSI_extreme_cross')

trading_decision = strategy_chanlun.run_direction_module_live()

print(trading_decision)

