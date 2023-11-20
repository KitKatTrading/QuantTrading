import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

import datetime
import pytz
import time

from binance import Client


datetime_format = "%Y-%m-%d %H:%M:%S+00:00"
api_key = 'W2E6HCVZOw02CFyLN1UenrnBSKpO2DJybasoPR51vUcl3ZqLP9HeTbF4yLWTjKKa'
api_secret = 'd4SWo9n2bGIZUhGfF1VAZQ1Yu6zJWnREQ8qd2517QX2m819ArNOsJgptQUNPz4xL'

###
def update_symbol_binance(name_symbol,time_scale,
                          datetime_format=datetime_format,
                          date_start='2020-01-01 00:00:00+00:00'):

    datetime_timenow_utc = datetime.datetime.utcnow()
    datetime_timenow_utc_timestamp = int(time.time()) * 1000
    print(f'Updating: {name_symbol}, {time_scale} || time now (UTC) = {datetime_timenow_utc}')

    ### Binance.US API setup
    api_key = 'jUm9MzGoyKGQFA8hFHBmSlYNQKdp2RI82wZ4gI70mNmjvagwoHd3r0Qm9t02VUMh'
    api_secret = 'z1Rh4ulilXvYskdnNlRczfJ5Ig9tBe5v2GRt2l1HQS8ODbGdFMOCVUkPQIfKnWvr'
    client = Client(api_key, api_secret)

    ### Datetime and interval configurations
    date_now_utc = datetime.datetime.utcnow()
    date_now_utc_pytz = date_now_utc.replace(tzinfo=pytz.utc)
    date_end_utc = datetime.datetime(date_now_utc.year, date_now_utc.month, date_now_utc.day + 1, 0, 0, 0)
    date_end_str = date_end_utc.strftime(datetime_format)

    if time_scale == '1w':
        name_binance_interval = Client.KLINE_INTERVAL_1WEEK
    elif time_scale == '1d':
        name_binance_interval = Client.KLINE_INTERVAL_1DAY
    elif time_scale == '12h':
        name_binance_interval = Client.KLINE_INTERVAL_12HOUR
    elif time_scale == '1h':
        name_binance_interval = Client.KLINE_INTERVAL_1HOUR
    elif time_scale == '5m':
        name_binance_interval = Client.KLINE_INTERVAL_5MINUTE


    ### Download data_raw from binance.us api
    # downalod data and add column names
    columns = ["Date", "Open", "High", "Low", "Close", "Volume",
               "Close time", "Quote asset volume", "Number of trades",
               "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    klines = client.futures_historical_klines(name_symbol,
                                             interval=name_binance_interval,
                                             start_str=date_start
                                              )
    data_binance_API = pd.DataFrame(klines, columns=columns)

    # get rid of the non-closed candles
    while data_binance_API.iloc[-1]['Close time'] > datetime_timenow_utc_timestamp:
        print('removing one row')
        data_binance_API = data_binance_API.iloc[:-1]

    # now clean up data_raw
    data_binance_API = data_binance_API[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    data_binance_API['Date'] = pd.to_datetime(data_binance_API['Date'], unit='ms')  # Convert the open "Date" column from Unix timestamps in milliseconds to human-readable date and time formats
    data_binance_API.set_index(['Date'], inplace=True)



    data_binance_API.index = data_binance_API.index.strftime(
        datetime_format + '+00:00')  # to match the time format in firebase storage
    data_binance_API.index = pd.to_datetime(data_binance_API.index)  # convert the index back to datetime format

    return data_binance_API

    # ### now download data_raw from firebase storage and check for missing dates
    # data_cloud = load_Y_OHLCV_csv_only(stock_full_name)
    # data_cloud['Date'] = pd.to_datetime(data_cloud['Date'])
    # data_cloud.set_index(['Date'], inplace=True)
    #
    # ### do the Boolean operation to get the missing candles that should be uploaded
    # # data_OHLC_upload = data_coinAPI.drop(data_cloud.index.intersection(data_coinAPI.index))
    # data_OHLC_upload = pd.concat([data_cloud, data_binance_API])
    # data_OHLC_upload = data_OHLC_upload[~data_OHLC_upload.index.duplicated(keep='last')]
    # data_OHLC_upload = data_OHLC_upload.iloc[:-1]  # get rid of the unclosed candlestick
    # # convert to '2023-05-03 15:59:59+00:00' string format
    # data_OHLC_upload.index = data_OHLC_upload.index.strftime('%Y-%m-%d %H:%M:%S') + '+00:00'
    #
    # ### ------- module_data upload -------- ###
    # # step 1: save the latest 120 closing points
    # # the end price only for the latest 120 points
    # pd_indicator_close = data_OHLC_upload['Close']
    # pd_indicator_close = pd_indicator_close.dropna()
    # pd_indicator_close = pd_indicator_close[-120:]
    # strategy_ref = db.reference('yList/'+stock_full_name)
    # strategy_ref.child('x_axis').set(list(pd.Series(pd_indicator_close.index.format())))
    # strategy_ref.child('y_axis').set(list(pd.Series(pd_indicator_close)))
    #
    # # step 2: save the file to storage (Y_OHLC)
    # token = uuid4()
    # bucket = storage.bucket()
    # blob = bucket.blob('stocks/' + stock_full_name + '.csv')
    # metadata = {"firebaseStorageDownloadTokens": token}
    # blob.metadata = metadata
    # pd_stock = data_OHLC_upload.dropna()  # the OCHLV data_raw
    # csv_stock_all = pd_stock.to_csv()
    # blob.upload_from_string(csv_stock_all, content_type='application/csv')
    # blob.make_public()
    #
    # # step 3: save the file to storage (Y_close only)
    # # target name for crypto 1h/1d close
    # if time_scale == '1d':
    #     stock_name_close_only = name_symbol
    # elif time_scale == '1h':
    #     stock_name_close_only = name_symbol[:-4] + '_1h'
    # data_CLOSE_upload = data_OHLC_upload[['Close']]
    # blob = bucket.blob('stocks/' + stock_name_close_only + '.csv')
    # metadata = {"firebaseStorageDownloadTokens": token}
    # blob.metadata = metadata
    # pd_stock = data_CLOSE_upload.dropna()  # the OCHLV data_raw
    # csv_stock_all = pd_stock.to_csv()
    # blob.upload_from_string(csv_stock_all, content_type='application/csv')
    # blob.make_public()


# if running this code as main
if __name__ == '__main__':
    data = update_symbol_binance('BTCUSDT', '1d')
    print(data)