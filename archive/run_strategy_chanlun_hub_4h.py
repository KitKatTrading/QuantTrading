from objects.strategy import Strategy

strategy_chanlun = Strategy(name_symbol='ALGOUSDT', data_source='binance', name_strategy='chanlun_12h',
                            timeframe_high='1d', timeframe_mid='1h', timeframe_low='1h',
                            function_high_timeframe='always_long',
                            function_mid_timeframe='chanlun',
                            function_low_timeframe='RSI_extreme_cross')

trading_decision = strategy_chanlun.check_ultimate_decision_all_modules()
direction_module_decision = strategy_chanlun.direction_module_decision
pattern_module_decision, fig_pattern = strategy_chanlun.pattern_module_decision
fig_pattern.show()

print(trading_decision)

