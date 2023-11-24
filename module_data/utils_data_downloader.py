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
    time_scale = args.time_scale  # Get the time scale from command line arguments

    name_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'MATICUSDT',
                    'AVAXUSDT', 'ATOMUSDT', 'UNIUSDT', 'APTUSDT', 'NEARUSDT', 'RUNEUSDT', 'OPUSDT', 'INJUSDT',
                    'LDOUSDT', 'EGLDUSDT', 'THETAUSDT', 'FTMUSDT', 'SANDUSDT', 'GALAUSDT', 'XTZUSDT', 'EOSUSDT',
                    'LTCUSDT', 'BCHUSDT', 'ZECUSDT', 'SEIUSDT', 'FILUSDT', 'DOTUSDT', 'LINKUSDT', 'AAVEUSDT']


    for name_symbol in name_symbols:
        print(f"Updating {name_symbol} {time_scale}")
        data = update_crypto_OHLC_from_binance(name_symbol, time_scale)



#
# import pandas as pd
#
#
# import datetime
# import pytz
# import time
#
# from binance import Client
#
#
# datetime_format = "%Y-%m-%d %H:%M:%S+00:00"
#
#
# ###
# def update_symbol_binance(name_symbol,time_scale,
#                           datetime_format=datetime_format,
#                           date_start='2020-01-01 00:00:00+00:00'):
#
#     datetime_timenow_utc = datetime.datetime.utcnow()
#     datetime_timenow_utc_timestamp = int(time.time()) * 1000
#     print(f'Updating: {name_symbol}, {time_scale} || time now (UTC) = {datetime_timenow_utc}')
#
#     ### Binance.US API setup
#     api_key = 'jUm9MzGoyKGQFA8hFHBmSlYNQKdp2RI82wZ4gI70mNmjvagwoHd3r0Qm9t02VUMh'
#     api_secret = 'z1Rh4ulilXvYskdnNlRczfJ5Ig9tBe5v2GRt2l1HQS8ODbGdFMOCVUkPQIfKnWvr'
#     client = Client(api_key, api_secret)
#
#     ### Datetime and interval configurations
#     date_now_utc = datetime.datetime.utcnow()
#     date_now_utc_pytz = date_now_utc.replace(tzinfo=pytz.utc)
#     date_end_utc = datetime.datetime(date_now_utc.year, date_now_utc.month, date_now_utc.day + 1, 0, 0, 0)
#     date_end_str = date_end_utc.strftime(datetime_format)
#
#     if time_scale == '1w':
#         type_binance_interval = Client.KLINE_INTERVAL_1WEEK
#     elif time_scale == '1d':
#         type_binance_interval = Client.KLINE_INTERVAL_1DAY
#     elif time_scale == '12h':
#         type_binance_interval = Client.KLINE_INTERVAL_12HOUR
#     elif time_scale == '1h':
#         type_binance_interval = Client.KLINE_INTERVAL_1HOUR
#     elif time_scale == '5m':
#         type_binance_interval = Client.KLINE_INTERVAL_5MINUTE
#
#
#     ### Download data_binance from binance.us api
#     # downalod data and add column names
#     columns = ["Date", "Open", "High", "Low", "Close", "Volume",
#                "Close time", "Quote asset volume", "Number of trades",
#                "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
#     klines = client.futures_historical_klines(name_symbol,
#                                               interval=type_binance_interval,
#                                               start_str=date_start,
#                                              )
#     data_binance_API = pd.DataFrame(klines, columns=columns)
#
#     # get rid of the non-closed candles
#     while data_binance_API.iloc[-1]['Close time'] > datetime_timenow_utc_timestamp:
#         print('removing one row')
#         data_binance_API = data_binance_API.iloc[:-1]
#
#     # now clean up data_binance
#     data_binance_API = data_binance_API[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
#     data_binance_API['Date'] = pd.to_datetime(data_binance_API['Date'], unit='ms')  # Convert the open "Date" column from Unix timestamps in milliseconds to human-readable date and time formats
#     data_binance_API.set_index(['Date'], inplace=True)
#     data_binance_API.index = data_binance_API.index.strftime(datetime_format)  # to match the time format in firebase storage
#     data_binance_API.index = pd.to_datetime(data_binance_API.index)  # convert the index back to datetime format
#
#     return data_binance_API
#
#     # ### now download data_binance from firebase storage and check for missing dates
#     # data_cloud = load_Y_OHLCV_csv_only(stock_full_name)
#     # data_cloud['Date'] = pd.to_datetime(data_cloud['Date'])
#     # data_cloud.set_index(['Date'], inplace=True)
#     #
#     # ### do the Boolean operation to get the missing candles that should be uploaded
#     # # data_OHLC_upload = data_coinAPI.drop(data_cloud.index.intersection(data_coinAPI.index))
#     # data_OHLC_upload = pd.concat([data_cloud, data_binance_API])
#     # data_OHLC_upload = data_OHLC_upload[~data_OHLC_upload.index.duplicated(keep='last')]
#     # data_OHLC_upload = data_OHLC_upload.iloc[:-1]  # get rid of the unclosed candlestick
#     # # convert to '2023-05-03 15:59:59+00:00' string format
#     # data_OHLC_upload.index = data_OHLC_upload.index.strftime('%Y-%m-%d %H:%M:%S') + '+00:00'
#     #
#     # ### ------- module_data upload -------- ###
#     # # step 1: save the latest 120 closing points
#     # # the end price only for the latest 120 points
#     # pd_indicator_close = data_OHLC_upload['Close']
#     # pd_indicator_close = pd_indicator_close.dropna()
#     # pd_indicator_close = pd_indicator_close[-120:]
#     # strategy_ref = db.reference('yList/'+stock_full_name)
#     # strategy_ref.child('x_axis').set(list(pd.Series(pd_indicator_close.index.format())))
#     # strategy_ref.child('y_axis').set(list(pd.Series(pd_indicator_close)))
#     #
#     # # step 2: save the file to storage (Y_OHLC)
#     # token = uuid4()
#     # bucket = storage.bucket()
#     # blob = bucket.blob('stocks/' + stock_full_name + '.csv')
#     # metadata = {"firebaseStorageDownloadTokens": token}
#     # blob.metadata = metadata
#     # pd_stock = data_OHLC_upload.dropna()  # the OCHLV data_binance
#     # csv_stock_all = pd_stock.to_csv()
#     # blob.upload_from_string(csv_stock_all, content_type='application/csv')
#     # blob.make_public()
#     #
#     # # step 3: save the file to storage (Y_close only)
#     # # target name for crypto 1h/1d close
#     # if time_scale == '1d':
#     #     stock_name_close_only = name_symbol
#     # elif time_scale == '1h':
#     #     stock_name_close_only = name_symbol[:-4] + '_1h'
#     # data_CLOSE_upload = data_OHLC_upload[['Close']]
#     # blob = bucket.blob('stocks/' + stock_name_close_only + '.csv')
#     # metadata = {"firebaseStorageDownloadTokens": token}
#     # blob.metadata = metadata
#     # pd_stock = data_CLOSE_upload.dropna()  # the OCHLV data_binance
#     # csv_stock_all = pd_stock.to_csv()
#     # blob.upload_from_string(csv_stock_all, content_type='application/csv')
#     # blob.make_public()
#
#
# # if running this code as main
# if __name__ == '__main__':
#     data = update_symbol_binance('BTCUSDT', '1d')
#     print(data)