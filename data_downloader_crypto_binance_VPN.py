import pandas as pd
import datetime
import os
import time
from binance import Client
import argparse  # Import argparse

# Setting pandas display options for better data visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# import local path
import config_local_path
PATH_DATA = config_local_path.gvars['dir_module_data_crypto_binance']

# import API keys
import config_binance_vpn
API_KEY = config_binance_vpn.gvars['API_KEY']
API_SECRET = config_binance_vpn.gvars['API_SECRET']

# Datetime formats and constants
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S+00:00"
DATETIME_START = '2020-01-01 00:00:00+00:00'

# Function to update symbol data from Binance
def initilize_crypto_OHLC_from_binance(symbol, time_scale):
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

    # Fetching data from Binance
    columns = ["Date", "Open", "High", "Low", "Close", "Volume",
               "Close time", "Quote asset volume", "Number of trades",
               "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    klines = client.futures_historical_klines(symbol, interval=type_binance_interval, start_str=DATETIME_START)
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
    csv_file_path = f'{PATH_DATA}/{symbol}_{time_scale}.csv'
    data_binance_API.to_csv(csv_file_path, index=True)

    return data_binance_API

def update_crypto_OHLC_from_binance(symbol, time_scale, name_subfolder, num_recent_candles=1000):
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

    # Path to the CSV file
    csv_file_path = f'{PATH_DATA}/{name_subfolder}/{symbol}_{time_scale}.csv'

    # Check if the file exists and get the last date
    if os.path.exists(csv_file_path):
        existing_data = pd.read_csv(csv_file_path)
        existing_data.set_index(['Date'], inplace=True)
        last_date = existing_data.index[-1]
    else:
        # If file does not exist, initialize the file with the start date
        last_date = DATETIME_START

    # Fetching recent data from Binance
    columns = ["Date", "Open", "High", "Low", "Close", "Volume",
               "Close time", "Quote asset volume", "Number of trades",
               "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    klines = client.futures_historical_klines(symbol, interval=type_binance_interval, start_str=last_date, limit=num_recent_candles)
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

    # Combine with existing data and remove duplicates
    if os.path.exists(csv_file_path):
        combined_data = pd.concat([existing_data, new_data_binance_API]).drop_duplicates()
    else:
        combined_data = new_data_binance_API

    # Save updated data back to CSV
    combined_data.to_csv(csv_file_path, index=True)

    return combined_data

def get_all_binance_future_symbols():
    client = Client(API_KEY, API_SECRET)
    futures_exchange_info = client.futures_exchange_info()
    symbols = [symbol['symbol'] for symbol in futures_exchange_info['symbols']]
    return symbols

# Running the function
if __name__ == '__main__':

    # Parse command line arguments for time scale
    parser = argparse.ArgumentParser(description='Download and update cryptocurrency data for a specific time scale.')
    parser.add_argument('time_scale', type=str, help='Time scale for the data, e.g., 1w, 1d, 12h, 1h')
    args = parser.parse_args()
    time_scale = args.time_scale  # Get the timescale from command line arguments
    # local run
    # time_scale = '1h'

    # get all binance future symbols
    name_symbols = get_all_binance_future_symbols()

    # remove ones not ending with "USDT"
    name_symbols = [name_symbol for name_symbol in name_symbols if name_symbol.endswith('USDT')]

    # update data for all symbols
    for name_symbol in name_symbols:
        print(f"Updating {name_symbol} {time_scale}")
        data = update_crypto_OHLC_from_binance(name_symbol, time_scale,
                                               name_subfolder='data_binance',
                                               num_recent_candles=1000)





    # name_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT',
    #                 'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT', 'NEARUSDT', 'RUNEUSDT', 'OPUSDT', 'INJUSDT',
    #                 'LDOUSDT', 'EGLDUSDT', 'THETAUSDT', 'FTMUSDT', 'SANDUSDT', 'GALAUSDT', 'XTZUSDT', 'EOSUSDT',
    #                 'LTCUSDT', 'BCHUSDT', 'ZECUSDT', 'SEIUSDT', 'FILUSDT', 'DOTUSDT', 'LINKUSDT', 'AAVEUSDT',
    #                 'OCEANUSDT', 'AGLDUSDT', 'TRBUSDT', 'ALICEUSDT', 'XMRUSDT', 'XLMUSDT',
    #                 'VETUSDT', 'SUSHIUSDT', 'KSMUSDT', 'GRTUSDT', '1INCHUSDT', 'ZENUSDT', 'YFIUSDT', 'BATUSDT',
    #                 'SNXUSDT', 'MKRUSDT', 'COMPUSDT', 'ENJUSDT', 'RENUSDT', 'CRVUSDT', 'MANAUSDT', 'MASKUSDT',
    #                 'CELRUSDT', 'OGNUSDT', 'REEFUSDT', 'DENTUSDT', 'RVNUSDT', 'DODOUSDT', 'HNTUSDT', 'TOMOUSDT',
    #                 'LITUSDT', 'COTIUSDT', 'AUDIOUSDT', 'AKROUSDT', 'CVCUSDT', 'STORJUSDT', 'HOTUSDT', 'NKNUSDT',
    #                 'WAVESUSDT', 'KAVAUSDT', 'ALGOUSDT', 'NEOUSDT', 'QTUMUSDT']
