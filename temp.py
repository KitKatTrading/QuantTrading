import time
current_time = int(time.time() * 1000)  # current time in milliseconds
start_time = current_time - (24 * 60 * 60 * 1000)  # 30 minutes before the current time
end_time = current_time

url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=30&startTime={start_time}&endTime={end_time}"

print(url)