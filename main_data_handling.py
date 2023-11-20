from module_data.DataHandler import DataHandlerStock

# Initialize DataHandler
data_handler = DataHandlerStock(path_root='D:\\QuantTradingData\\Stocks')

#
# symbols = ['TSLA', 'AAPL', 'MSFT', 'SPY', 'QQQ']
symbols = ['TSLA']
intervals = ['1d', '1h', '10m']

for symbol in symbols:
    print(f"Updating data for {symbol}...")

    # Update and save data_raw for different intervals.
    for interval in intervals:
        print(f"Updating data for {symbol} at interval {interval}...")

        data_handler.update_data(symbol='TSLA', interval=interval)


