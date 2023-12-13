from Objects.strategy import Strategy

strategy_chanlun_12h = Strategy(name_symbol='OPUSDT', data_source='binance', name_strategy='chanlun_12h',
                                timeframe_high='1d', timeframe_mid='4h', timeframe_low='1h',
                                function_high_timeframe='SMA_5_10_20_trend',
                                function_mid_timeframe='chanlun_central_hub',
                                function_low_timeframe='RSI_divergence')

trading_decision = strategy_chanlun_12h.check_trading_decision_all_modules()
print(trading_decision)

