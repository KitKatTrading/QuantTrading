import pandas as pd
import datetime
import os
import time
from binance import Client
import argparse  # Import argparse

# Setting pandas display options for better data visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Constants and API Credentials
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S+00:00"
DATETIME_START = '2020-01-01 00:00:00+00:00'
API_KEY = 'W2E6HCVZOw02CFyLN1UenrnBSKpO2DJybasoPR51vUcl3ZqLP9HeTbF4yLWTjKKa'
API_SECRET = 'd4SWo9n2bGIZUhGfF1VAZQ1Yu6zJWnREQ8qd2517QX2m819ArNOsJgptQUNPz4xL'
PATH_DATA = 'data_binance'

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

def update_crypto_OHLC_from_binance(symbol, time_scale, num_recent_candles=100):
    client = Client(API_KEY, API_SECRET)

    # Mapping time_scale to binance interval
    interval_map = {
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '5m': Client.KLINE_INTERVAL_5MINUTE
    }
    type_binance_interval = interval_map.get(time_scale)

    # Path to the CSV file
    csv_file_path = f'{PATH_DATA}/{symbol}_{time_scale}.csv'

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


class DataHandlerStock:
    def __init__(self,
                 path_root,
                 datetime_start="2010-01-01"):
        self.path_root = path_root
        self.datetime_start = datetime_start

    # def fetch_data_yahoo_finance(self, symbol, start, end, interval):
    #     """
    #     Fetch historical data_binance for the given symbol, start, end, and interval.
    #     """
    #     stock_data = yf.download(symbol, start=start, end=end, interval=interval)
    #     stock_data.reset_index(inplace=True)
    #     return stock_data

    def save_to_csv(self, df, file_path):
        """
        Save the DataFrame to a local CSV file.
        """
        df.to_csv(file_path, index=False)

    def read_from_csv(self, file_path):
        """
        Read the DataFrame from a local CSV file.
        """
        if os.path.exists(file_path):
            return pd.read_csv(file_path, parse_dates=['Datetime'])
        return None

    def update_data(self, symbol, interval):
        """
        Update the local CSV file with the latest historical data_binance.
        """
        file_path = f"{symbol}_{interval}.csv"
        existing_data = self.read_from_csv(file_path)

        if existing_data is not None:
            # If data_binance exists, update the missing rows.
            start_date = existing_data['Datetime'].max()
            new_data = self.fetch_data_yahoo_finance(symbol, start=start_date, end=pd.to_datetime("today"), interval=interval)
            updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['Datetime'])
        else:
            # If no data_binance exists, fetch all available data_binance.
            updated_data = self.fetch_data_yahoo_finance(symbol, start="2010-01-01", end=pd.to_datetime("today"), interval=interval)

        # Save the updated data_binance to CSV.
        self.save_to_csv(updated_data, file_path)


# Running the function
if __name__ == '__main__':
    # Parse command line arguments for time scale
    parser = argparse.ArgumentParser(description='Download and update cryptocurrency data for a specific time scale.')
    parser.add_argument('time_scale', type=str, help='Time scale for the data, e.g., 1w, 1d, 12h, 1h')
    args = parser.parse_args()
    time_scale = args.time_scale  # Get the timescale from command line arguments

    name_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT',
                    'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT', 'NEARUSDT', 'RUNEUSDT', 'OPUSDT', 'INJUSDT',
                    'LDOUSDT', 'EGLDUSDT', 'THETAUSDT', 'FTMUSDT', 'SANDUSDT', 'GALAUSDT', 'XTZUSDT', 'EOSUSDT',
                    'LTCUSDT', 'BCHUSDT', 'ZECUSDT', 'SEIUSDT', 'FILUSDT', 'DOTUSDT', 'LINKUSDT', 'AAVEUSDT',
                    'OCEANUSDT', 'AGLDUSDT', 'TRBUSDT', 'ALICEUSDT', 'LUNAUSDT', 'ICPUSDT', 'XMRUSDT', 'XLMUSDT',
                    'VETUSDT', 'SUSHIUSDT', 'KSMUSDT', 'GRTUSDT', '1INCHUSDT', 'ZENUSDT', 'YFIUSDT', 'BATUSDT',
                    'SNXUSDT', 'MKRUSDT', 'COMPUSDT', 'ENJUSDT', 'RENUSDT', 'CRVUSDT', 'BALUSDT', 'SKLUSDT',
                    'MANAUSDT', 'MASKUSDT', 'AXSUSDT', 'SXPUSDT', 'BANDUSDT', 'IOSTUSDT', 'CELRUSDT', 'OGNUSDT',
                    'REEFUSDT', 'DENTUSDT', 'RVNUSDT', 'DODOUSDT', 'HNTUSDT', 'CTKUSDT', 'TOMOUSDT', 'LITUSDT',
                    'COTIUSDT', 'AUDIOUSDT', 'AKROUSDT', 'CVCUSDT', 'STORJUSDT', 'BTSUSDT', 'ARDRUSDT', 'SCUSDT',
                    'DGBUSDT', 'STMXUSDT', 'HOTUSDT', 'DNTUSDT', 'ARDRUSDT', 'SCUSDT', 'DGBUSDT', 'STMXUSDT',
                    'HOTUSDT', 'DNTUSDT', 'NKNUSDT', 'BLZUSDT', 'RLCUSDT', 'SNTUSDT', 'WAVESUSDT', 'LSKUSDT',
                    'XEMUSDT', 'ZILUSDT', 'ZRXUSDT', 'REPUSDT', 'KAVAUSDT', 'ALGOUSDT', 'NEOUSDT', 'QTUMUSDT',
                    'IOTAUSDT', 'BANDUSDT', 'NANOUSDT', 'ONTUSDT', 'OMGUSDT', 'ZRXUSDT', 'REPUSDT', 'ALGOUSDT',
                    'NEOUSDT', 'QTUMUSDT', 'IOTAUSDT', 'BANDUSDT', 'NANOUSDT', 'ONTUSDT', 'OMGUSDT', 'ZRXUSDT']

    # check if there are repeated entries
    # assert len(name_symbols) == len(set(name_symbols)), "There are repeated entries in name_symbols"

    for name_symbol in name_symbols:
        print(f"Updating {name_symbol} {time_scale}")
        data = update_crypto_OHLC_from_binance(name_symbol, time_scale)


