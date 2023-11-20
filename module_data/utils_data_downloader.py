import pandas as pd
import config_data_downloader
import datetime

from binance import Client


### Configs
datetime_format = config_data_downloader.gvars['datetime_format']
api_key = config_data_downloader.gvars['api_key']
api_secret = config_data_downloader.gvars['api_secret']

###
def update_symbol_binance(name_symbol,time_scale, date_start='2020-01-01 00:00:00+00:00'):

    print(f'Uploading: {name_symbol}, {time_scale}')

    ### Binance.US API setup
    api_key = 'jUm9MzGoyKGQFA8hFHBmSlYNQKdp2RI82wZ4gI70mNmjvagwoHd3r0Qm9t02VUMh'
    api_secret = 'z1Rh4ulilXvYskdnNlRczfJ5Ig9tBe5v2GRt2l1HQS8ODbGdFMOCVUkPQIfKnWvr'
    client = Client(api_key, api_secret)

    # coinAPI data download configurations
    date_format = "%Y-%m-%d %H:%M:%S"
    # date_now_utc = datetime.datetime.utcnow()
    # date_now_utc_pytz = date_now_utc.replace(tzinfo=pytz.utc)
    # date_end_utc = datetime.datetime(date_now_utc.year, date_now_utc.month, date_now_utc.day + 1, 0, 0, 0)
    # date_end_str = date_end_utc.strftime(date_format)

    ### Download data from binance.us api
    columns = ["Date", "Open", "High", "Low", "Close", "Volume",
               "Close time", "Quote asset volume", "Number of trades",
               "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
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
    klines = client.get_historical_klines(name_symbol,
                                         interval=Client.KLINE_INTERVAL_12HOUR,
                                         start_str=date_start,)
    data_binance_API = pd.DataFrame(klines, columns=columns)
    # only keep candles that are closed (closed if timenow > supposed candle close time "time_period_end"
    # data_coinAPI = data_coinAPI[
    #    pd.to_datetime(data_coinAPI['time_period_end']).dt.to_pydatetime() < date_now_utc_pytz]

    # now clean up data
    data_binance_API = data_binance_API[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    data_binance_API['Date'] = pd.to_datetime(data_binance_API['Date'], unit='ms')  # Convert the open "Date" column from Unix timestamps in milliseconds to human-readable date and time formats
    data_binance_API.set_index(['Date'], inplace=True)
    # based on discussion on 2022-12-28, use closing timestamp instead of open timestamp to upload daily data
    data_binance_API.index = data_binance_API.index
    data_binance_API.index = data_binance_API.index.strftime(
        date_format + '+00:00')  # to match the time format in firebase storage
    data_binance_API.index = pd.to_datetime(data_binance_API.index)  # convert the index back to datetime format

    return data_binance_API

    # ### now download data from firebase storage and check for missing dates
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
    # pd_stock = data_OHLC_upload.dropna()  # the OCHLV data
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
    # pd_stock = data_CLOSE_upload.dropna()  # the OCHLV data
    # csv_stock_all = pd_stock.to_csv()
    # blob.upload_from_string(csv_stock_all, content_type='application/csv')
    # blob.make_public()


# if running this code as main
if __name__ == '__main__':
    data = update_symbol_binance('BTCUSDT', '1d')
    print(data)