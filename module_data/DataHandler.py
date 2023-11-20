import os
import pandas as pd
import yfinance as yf
import configparser
import config
import datetime
import pytz
import plotly.graph_objects as go

### local debug view config
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


""" Class DataHandler
### This class is responsible for fetching, saving, and reading historical data_raw.
"""


class DataHandlerStock:
    def __init__(self,
                 path_root,
                 datetime_start="2010-01-01"):
        self.path_root = path_root
        self.datetime_start = datetime_start

    def fetch_data_yahoo_finance(self, symbol, start, end, interval):
        """
        Fetch historical data_raw for the given symbol, start, end, and interval.
        """
        stock_data = yf.download(symbol, start=start, end=end, interval=interval)
        stock_data.reset_index(inplace=True)
        return stock_data

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
        Update the local CSV file with the latest historical data_raw.
        """
        file_path = f"{symbol}_{interval}.csv"
        existing_data = self.read_from_csv(file_path)

        if existing_data is not None:
            # If data_raw exists, update the missing rows.
            start_date = existing_data['Datetime'].max()
            new_data = self.fetch_data_yahoo_finance(symbol, start=start_date, end=pd.to_datetime("today"), interval=interval)
            updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['Datetime'])
        else:
            # If no data_raw exists, fetch all available data_raw.
            updated_data = self.fetch_data_yahoo_finance(symbol, start="2010-01-01", end=pd.to_datetime("today"), interval=interval)

        # Save the updated data_raw to CSV.
        self.save_to_csv(updated_data, file_path)


