import pandas as pd
import datetime
import time
import argparse
# from objects.strategy import Strategy
from binance import Client
from discordwebhook import Discord
import importlib
# Setting pandas display options for better data visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# import API keys
from archive import config_binance_vpn

API_KEY = config_binance_vpn.gvars['API_KEY']
API_SECRET = config_binance_vpn.gvars['API_SECRET']

# Datetime formats and constants
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S+00:00"

# import Discord webhook
from config import config_discord_channels

webhook_all = {'15m': config_discord_channels.gvars['url_webhook_powaysha_15m'],
               '1h': config_discord_channels.gvars['url_webhook_powaysha_1h'],
               '4h': config_discord_channels.gvars['url_webhook_powaysha_4h'],
               '1d': config_discord_channels.gvars['url_webhook_powaysha_1d'],
               }

# Create start date string for Binance API call that is always 1000 candles before now
datetime_now = datetime.datetime.utcnow()
dict_datetime_start_binance = {'5m': datetime_now - datetime.timedelta(hours=1000 / 12),
                               '15m': datetime_now - datetime.timedelta(hours=1000 / 4),
                               '30m': datetime_now - datetime.timedelta(hours=1000 / 2),
                               '1h': datetime_now - datetime.timedelta(hours=1000),
                               '2h': datetime_now - datetime.timedelta(hours=1000 * 2),
                               '4h': datetime_now - datetime.timedelta(hours=1000 * 4),
                               '1d': datetime_now - datetime.timedelta(hours=1000 * 24),
                               }

# Mapping time_scale to binance interval
dict_interval_map = {
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

def get_crypto_OHLC_from_binance_API(symbol, time_scale, num_recent_candles=1000):

    """ Always only get the most recent 1000 candles"""

    # set up the client
    client = Client(API_KEY, API_SECRET)

    # get constants
    type_binance_interval = dict_interval_map.get(time_scale)
    datetime_start = dict_datetime_start_binance[time_scale].strftime(DATETIME_FORMAT)

    # Fetching recent data from Binance
    columns = ["Date", "Open", "High", "Low", "Close", "Volume",
               "Close time", "Quote asset volume", "Number of trades",
               "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    klines = client.futures_historical_klines(symbol,
                                              interval=type_binance_interval,
                                              start_str=datetime_start,
                                              limit=num_recent_candles)
    new_data_binance_API = pd.DataFrame(klines, columns=columns)

    # Removing non-closed candles
    current_time_milliseconds = int(time.time()) * 1000
    new_data_binance_API = new_data_binance_API[new_data_binance_API['Close time'] < current_time_milliseconds]

    # Processing and cleaning new data
    new_data_binance_API = new_data_binance_API[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    new_data_binance_API['Date'] = pd.to_datetime(new_data_binance_API['Date'], unit='ms')
    new_data_binance_API.set_index(['Date'], inplace=True)
    new_data_binance_API.index = new_data_binance_API.index.strftime(DATETIME_FORMAT)
    new_data_binance_API.index = pd.to_datetime(new_data_binance_API.index)

    return new_data_binance_API

def divide_chunks(lst, num):
    avg = len(lst) / float(num)
    out = []
    last = 0.0
    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg
    return out

def get_all_binance_future_symbols(num_total_partitions: int, partition_id: int):

    # get all symbols
    client = Client(API_KEY, API_SECRET)
    futures_exchange_info = client.futures_exchange_info()
    symbols = [symbol['symbol'] for symbol in futures_exchange_info['symbols']]

    # re-order the symbols aphabetically
    symbols.sort()

    # get only the partition requested
    print("\n\n", partition_id, ' partition of ', num_total_partitions, ' no of partitions')

    sublist_strategies = divide_chunks(symbols, num_total_partitions)
    crypto_list_partition = sublist_strategies[partition_id - 1]

    return crypto_list_partition

# Running the function
if __name__ == '__main__':

    # # # set timescale
    # time_scale = '1h'  # Get the timescale from command line arguments
    # num_total_partitions = 5
    # partition_id = 1

    # read command line arguments to get time_scale, num_total_partitions, partition_id
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_scale', type=str, default='15m')
    parser.add_argument('--num_total_partitions', type=int, default=20)
    parser.add_argument('--partition_id', type=int, default=1)
    args = parser.parse_args()
    time_scale = args.time_scale
    num_total_partitions = args.num_total_partitions
    partition_id = args.partition_id

    # constants and params
    datetime_format = '%Y-%m-%d %H:%M:%S+00:00'
    datetime_start = dict_datetime_start_binance[time_scale]

    # import function
    strategy_mid_timeframe = importlib.import_module(f"module_pattern.chanlun_poway")

    # set up discord webhook
    webhook_kk_quant_discord_powaysha = webhook_all[time_scale]
    webhook_discord = Discord(url=webhook_kk_quant_discord_powaysha)

    # get all binance future symbols
    name_symbols = get_all_binance_future_symbols(num_total_partitions=num_total_partitions, partition_id=partition_id)

    for name_symbol in name_symbols:

        # First updating data
        print(f"Updating {name_symbol} {time_scale}")
        data = get_crypto_OHLC_from_binance_API(name_symbol, time_scale, num_recent_candles=1000)

        # Then check signals
        data_input = data.iloc[-500:]
        pattern_module_decision, fig = strategy_mid_timeframe.main(data,
                                                                   name_symbol=name_symbol,
                                                                   time_frame=time_scale)

        # initialize trade direction
        trade_direction = 'None'
        if pattern_module_decision == 1:
            trade_direction = 'Long'
        elif pattern_module_decision == -1:
            trade_direction = 'Short'

        # broadcast to discord
        if pattern_module_decision != 0:

            #
            # fig.show()

            # set up datetime
            datetime_now = datetime.datetime.utcnow()
            datetime_now_str = datetime_now.strftime(datetime_format)

            # set up message
            message_separator = '-------------------------------------\n'
            message_time = f'时间 {datetime_now_str}\n'
            message_name = f'标的 {name_symbol}\n'
            message_timescale = f'周期 {time_scale}\n'
            message_direction = f'方向 {trade_direction}\n'
            message_all = message_separator + message_time + message_name + message_timescale + message_direction

            # send message
            webhook_discord.post(content=message_all)



