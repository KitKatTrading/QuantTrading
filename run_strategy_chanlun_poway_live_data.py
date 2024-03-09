from Objects.strategy_live_data import Strategy
from binance import Client
from discordwebhook import Discord
import pandas as pd
import time
import datetime
import argparse  # Import argparse

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S+00:00"
def get_all_binance_future_symbols(API_KEY, API_SECRET):
    client = Client(API_KEY, API_SECRET)
    futures_exchange_info = client.futures_exchange_info()
    symbols = [symbol['symbol'] for symbol in futures_exchange_info['symbols']]
    return symbols

def get_crypto_OHLC_from_binance_live(symbol, time_scale, num_recent_candles=1000):

    # initialize client
    client = Client(API_KEY, API_SECRET)

    # Mapping time_scale to binance interval
    interval_map = {
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE
    }
    type_binance_interval = interval_map.get(time_scale)

    # Mapping the time_scale to minutes
    time_scale_map = {
        '1w': 7 * 24 * 60,
        '1d': 24 * 60,
        '12h': 12 * 60,
        '8h': 8 * 60,
        '4h': 4 * 60,
        '1h': 60,
        '30m': 30,
        '15m': 15,
        '5m': 5
    }
    num_minutes_per_candle = time_scale_map.get(time_scale)
    current_time_milliseconds = int(time.time()) * 1000
    start_time_milliseconds = current_time_milliseconds - num_recent_candles * num_minutes_per_candle * 60 * 1000
    klines = client.futures_historical_klines(symbol,
                                              interval=type_binance_interval,
                                              start_str=start_time_milliseconds,
                                              limit=num_recent_candles)
    # Fetching data from Binance
    columns = ["Date", "Open", "High", "Low", "Close", "Volume",
               "Close time", "Quote asset volume", "Number of trades",
               "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    data_binance_API = pd.DataFrame(klines, columns=columns)

    # Removing non-closed candles
    current_time_milliseconds = int(time.time()) * 1000
    data_binance_API = data_binance_API[data_binance_API['Close time'] <= current_time_milliseconds]

    # Processing and cleaning data
    data_binance_API = data_binance_API[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    data_binance_API['Date'] = pd.to_datetime(data_binance_API['Date'], unit='ms')  # Convert the open "Date" column from Unix timestamps in milliseconds to human-readable date and time formats
    data_binance_API.set_index(['Date'], inplace=True)
    data_binance_API.index = data_binance_API.index.strftime(DATETIME_FORMAT)  # to match the time format in firebase storage
    data_binance_API.index = pd.to_datetime(data_binance_API.index)  # convert the index back to datetime format

    # Save data to CSV
    # csv_file_path = f'{PATH_DATA}/{symbol}_{time_scale}.csv'
    # data_binance_API.to_csv(csv_file_path, index=True)

    return data_binance_API


if __name__ == '__main__':

    # Parse command line arguments for time scale
    parser = argparse.ArgumentParser(description='Download and update cryptocurrency data for a specific time scale.')
    parser.add_argument('time_scale', type=str, help='Time scale for the data, e.g., 1w, 1d, 12h, 1h')
    args = parser.parse_args()
    time_scale = args.time_scale  # Get the timescale from command line arguments

    # Get all future symbol names
    import config_binance_vpn
    API_KEY = config_binance_vpn.gvars['API_KEY']
    API_SECRET = config_binance_vpn.gvars['API_SECRET']
    name_symbols = get_all_binance_future_symbols(API_KEY, API_SECRET)

    # Remove symbols not ending with "USDT"
    name_symbols = [name_symbol for name_symbol in name_symbols if name_symbol[-4:] == 'USDT']

    # Single symbol test
    # name_symbols = ['BTCUSDT', 'ETHUSDT']

    # run all symbols
    for name_symbol in name_symbols:

        print(name_symbol)

        # download data live
        t1 = time.time()
        data = get_crypto_OHLC_from_binance_live(symbol=name_symbol, time_scale=time_scale, num_recent_candles=1000)
        t2 = time.time()
        print(f'Time to download data: {t2 - t1} seconds')


        # run strategy
        strategy_chanlun = Strategy(name_symbol=name_symbol, data_source='binance', name_strategy='chanlun_poway',
                                    timeframe_high=time_scale, timeframe_mid=time_scale, timeframe_low=time_scale,
                                    function_high_timeframe='always_long',
                                    function_mid_timeframe='chanlun_poway',
                                    function_low_timeframe='RSI_pinbar',
                                    df_OHLC_high=data, df_OHLC_mid=data, df_OHLC_low=data,
                                    )

        # run the strategy - do this first
        trading_decision = strategy_chanlun.check_ultimate_decision_all_modules()

        # get decision
        decision_direction = strategy_chanlun.decision_direction
        decision_pattern = strategy_chanlun.decision_pattern
        decision_entry = strategy_chanlun.decision_entry

        # initialize trade direction
        trade_direction = 'None'
        if decision_pattern == 1:
            trade_direction = 'Long'
        elif decision_pattern == -1:
            trade_direction = 'Short'

            # save image
            # broadcast to discord
        if trading_decision != 0:

            # get pattern and entry charts
            fig_pattern = strategy_chanlun.chart_pattern
            fig_entry = strategy_chanlun.chart_entry

            # set up datetime
            datetime_now = datetime.datetime.utcnow()
            datetime_now_str = datetime_now.strftime(DATETIME_FORMAT)

            # set up message
            message_separator = '-------------------------------------\n'
            message_time = f'时间 {datetime_now_str}\n'
            message_name = f'标的 {name_symbol}\n'
            message_timescale = f'周期 {time_scale}\n'
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

            # send pattern to discord fig_pattern
            fig_pattern.write_image('fig_pattern.png')
            webhook_discord.post(
                file={
                    "file1": open("fig_pattern.png", "rb"),
                },
            )
            import os

            os.remove('fig_pattern.png')

            # send image fig_entry
            fig_entry.write_image('fig_entry.png')
            webhook_discord.post(
                file={
                    "file1": open("fig_entry.png", "rb"),
                },
            )
            import os

            os.remove('fig_entry.png')

        # check total run time
        t3 = time.time()
        print(f'Time to complete analysis: {t3 - t1} seconds')