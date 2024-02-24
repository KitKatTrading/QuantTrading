from Objects.strategy import Strategy
from binance import Client
from discordwebhook import Discord
import datetime

def get_all_binance_future_symbols(API_KEY, API_SECRET):
    client = Client(API_KEY, API_SECRET)
    futures_exchange_info = client.futures_exchange_info()
    symbols = [symbol['symbol'] for symbol in futures_exchange_info['symbols']]
    return symbols



if __name__ == '__main__':

    # Strategy parameter setting ###
    time_frame_mid = '1h'

    # datetime format
    datetime_format = '%Y-%m-%d %H:%M:%S+00:00'

    # # Get all future symbol names
    import config_binance_vpn
    API_KEY = config_binance_vpn.gvars['API_KEY']
    API_SECRET = config_binance_vpn.gvars['API_SECRET']
    name_symbols = get_all_binance_future_symbols(API_KEY, API_SECRET)

    # single symbol test
    # name_symbols = ['BTCUSDT', 'ETHUSDT']

    # run all symbols
    for name_symbol in name_symbols:

        print(name_symbol)
        strategy_chanlun = Strategy(name_symbol=name_symbol, data_source='binance', name_strategy='chanlun_poway',
                                    timeframe_high=time_frame_mid, timeframe_mid=time_frame_mid, timeframe_low=time_frame_mid,
                                    function_high_timeframe='always_long',
                                    function_mid_timeframe='chanlun_poway',
                                    function_low_timeframe='RSI_pinbar'
                                    )

        # run the strategy - do this first
        trading_decision = strategy_chanlun.check_ultimate_decision_all_modules()

        # get decision
        decision_direction = strategy_chanlun.decision_direction
        decision_pattern, fig_pattern = strategy_chanlun.decision_pattern
        decision_entry, fig_entry = strategy_chanlun.decision_entry
        # fig_entry.show()
        # fig_pattern.show()

        # initialize trade direction
        trade_direction = 'None'
        if decision_pattern == 1:
            trade_direction = 'Long'
        elif decision_pattern == -1:
            trade_direction = 'Short'

        # save image
        # broadcast to discord
        if trading_decision != 0:

            # set up datetime
            datetime_now = datetime.datetime.utcnow()
            datetime_now_str = datetime_now.strftime(datetime_format)

            # set up message
            message_separator = '-------------------------------------\n'
            message_time = f'时间 {datetime_now_str}\n'
            message_name = f'标的 {name_symbol}\n'
            message_timescale = f'周期 {time_frame_mid}\n'
            message_direction = f'方向 {trade_direction}\n'

            # set up discord webhook
            webhook_kk_quant_discord_powaysha = 'https://discord.com/api/webhooks/1187690857432895588/rLiDFkbL2pPUmDnaccHDzXLpr4KD5wBTwuy78zL0QaidIWlBsdlKMU_jWxpag9azXKiL'
            webhook_discord = Discord(url=webhook_kk_quant_discord_powaysha)

            # send message
            webhook_discord.post(content=message_separator)
            webhook_discord.post(content=message_time)
            webhook_discord.post(content=message_name)
            webhook_discord.post(content=message_timescale)
            webhook_discord.post(content=message_direction)

            # send image fig_entry
            fig_entry.write_image('fig_entry.png')
            webhook_discord.post(
                file={
                    "file1": open("fig_entry.png", "rb"),
                },
            )

            # remove image
            import os
            os.remove('fig_entry.png')




