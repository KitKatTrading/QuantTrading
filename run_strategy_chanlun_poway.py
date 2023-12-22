from Objects.strategy import Strategy
from discordwebhook import Discord
import datetime

# alert setup
name_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT',
                'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT', 'NEARUSDT', 'RUNEUSDT', 'OPUSDT', 'INJUSDT',
                'LDOUSDT', 'EGLDUSDT', 'THETAUSDT', 'FTMUSDT', 'SANDUSDT', 'GALAUSDT', 'XTZUSDT', 'EOSUSDT',
                'LTCUSDT', 'BCHUSDT', 'ZECUSDT', 'SEIUSDT', 'FILUSDT', 'DOTUSDT', 'LINKUSDT', 'AAVEUSDT',
                'OCEANUSDT', 'AGLDUSDT', 'TRBUSDT', 'ALICEUSDT', 'XMRUSDT', 'XLMUSDT',
                'VETUSDT', 'SUSHIUSDT', 'KSMUSDT', 'GRTUSDT', '1INCHUSDT', 'ZENUSDT', 'YFIUSDT', 'BATUSDT',
                'SNXUSDT', 'MKRUSDT', 'COMPUSDT', 'ENJUSDT', 'RENUSDT', 'CRVUSDT', 'MANAUSDT', 'MASKUSDT',
                'CELRUSDT', 'OGNUSDT', 'REEFUSDT', 'DENTUSDT', 'RVNUSDT', 'DODOUSDT', 'HNTUSDT', 'TOMOUSDT',
                'LITUSDT', 'COTIUSDT', 'AUDIOUSDT', 'AKROUSDT', 'CVCUSDT', 'STORJUSDT', 'HOTUSDT', 'NKNUSDT',
                'WAVESUSDT', 'KAVAUSDT', 'ALGOUSDT', 'NEOUSDT', 'QTUMUSDT']
time_frame_mid = '1h'

# Set up datetime format
datetime_format = '%Y-%m-%d %H:%M:%S+00:00'

# set up discord webhook
webhook_kk_quant_discord_powaysha = 'https://discord.com/api/webhooks/1187690857432895588/rLiDFkbL2pPUmDnaccHDzXLpr4KD5wBTwuy78zL0QaidIWlBsdlKMU_jWxpag9azXKiL'
webhook_discord = Discord(url=webhook_kk_quant_discord_powaysha)

# run all symbols
for name_symbol in name_symbols:

    print(name_symbol)

    strategy_chanlun = Strategy(name_symbol=name_symbol, data_source='binance', name_strategy='chanlun_poway',
                                timeframe_high=time_frame_mid, timeframe_mid=time_frame_mid, timeframe_low=time_frame_mid,
                                function_high_timeframe='always_long',
                                function_mid_timeframe='chanlun_poway',
                                function_low_timeframe='RSI_extreme_cross')
    trading_decision = strategy_chanlun.check_ultimate_decision_all_modules()
    direction_module_decision = strategy_chanlun.direction_module_decision
    pattern_module_decision, fig_pattern = strategy_chanlun.pattern_module_decision

    # initialize trade direction
    trade_direction = 'None'
    if pattern_module_decision == 1:
        trade_direction = 'Long'
    elif pattern_module_decision == -1:
        trade_direction = 'Short'

    # save image
    # broadcast to discord
    if pattern_module_decision != 0:

        # set up datetime
        datetime_now = datetime.datetime.utcnow()
        datetime_now_str = datetime_now.strftime(datetime_format)

        # set up message
        message_separator = '-------------------------------------\n'
        message_time = f'时间 {datetime_now_str}\n'
        message_name = f'标的 {name_symbol}\n'
        message_timescale = f'周期 {time_frame_mid}\n'
        message_direction = f'方向 {trade_direction}\n'

        # send message
        webhook_discord.post(content=message_separator)
        webhook_discord.post(content=message_time)
        webhook_discord.post(content=message_name)
        webhook_discord.post(content=message_timescale)
        webhook_discord.post(content=message_direction)




