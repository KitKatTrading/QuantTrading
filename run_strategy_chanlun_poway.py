from Objects.strategy import Strategy
from binance import Client
from discordwebhook import Discord
import datetime

def get_all_binance_future_symbols(API_KEY, API_SECRET):
    client = Client(API_KEY, API_SECRET)
    futures_exchange_info = client.futures_exchange_info()
    symbols = [symbol['symbol'] for symbol in futures_exchange_info['symbols']]
    return symbols

# alert setup time frame
time_frame_mid = '15m'

# Set up datetime format
datetime_format = '%Y-%m-%d %H:%M:%S+00:00'

# set up discord webhook
webhook_kk_quant_discord_powaysha = 'https://discord.com/api/webhooks/1187690857432895588/rLiDFkbL2pPUmDnaccHDzXLpr4KD5wBTwuy78zL0QaidIWlBsdlKMU_jWxpag9azXKiL'
webhook_discord = Discord(url=webhook_kk_quant_discord_powaysha)

# # Get all future symbol names
import config_binance_vpn
API_KEY = config_binance_vpn.gvars['API_KEY']
API_SECRET = config_binance_vpn.gvars['API_SECRET']
name_symbols = get_all_binance_future_symbols(API_KEY, API_SECRET)

# single symbol test
# name_symbols = ['OPUSDT']

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
    fig_pattern.show()

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




