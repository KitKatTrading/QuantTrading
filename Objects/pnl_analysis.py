import pandas as pd
import os

dir_data = 'module_data'
dir_backtesting = 'module_backtesting'

# Setting pandas display options for better data visibility
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

class PNL:

    def __init__(self, job_name, datetime_bt_start, datetime_bt_end, timeframe_low):

        self.bt_start = datetime_bt_start
        self.bt_end = datetime_bt_end
        self.timeframe_low = timeframe_low
        self.dir_backtesting_job = os.path.join(dir_backtesting, job_name)
        self.df_trade_log_all = None
        self.dict_df_ohlc = None
        self.df_pnl_curve = None

        ### Combine all individual trade log files into one master trade log file
        # generate the master trade log file
        df_trade_log_all = pd.DataFrame()

        # generate a dict of dataframes for each symbols OHLC data
        dict_df_ohlc = {}

        # get all the trade log files
        # the trade log file should be named as e.g.:'df_trade_log_chanlun20231216-234041_DOTUSDT.csv
        for file in os.listdir(self.dir_backtesting_job):
            if file.endswith(".csv") and file.startswith("df_trade_log_"):

                # read the file
                df_trade_log = pd.read_csv(os.path.join(self.dir_backtesting_job, file), index_col=0)

                # add the name_symbol column
                name_symbol = file.split('_')[-1].split('.')[0]
                df_trade_log['name_symbol'] = name_symbol

                # add the file to the master trade log file and update index
                df_trade_log_all = pd.concat([df_trade_log_all, df_trade_log]).reset_index(drop=True)

                # add the OHLC data to the dict
                df_ohlc = pd.read_csv(os.path.join(dir_data, 'data_binance', f'{name_symbol}_{self.timeframe_low}.csv'), index_col=0)
                dict_df_ohlc[name_symbol] = df_ohlc


        # clean up the master trade log file
        df_trade_log_all.drop(columns=['entry_number', 'entry_idx', 'exit_idx', 'max_profit', 'pnl_max', 'rrr', 'rrr_max',
                                        'duration', 'win_loss'], inplace=True)

        # re-order the rows by datetime_entry in an ascending order to simulate the trades chronologically.
        df_trade_log_all.sort_values(by=['datetime_entry'], inplace=True)

        # save the dataframes to class attributes
        self.df_trade_log_all = df_trade_log_all
        self.dict_df_ohlc = dict_df_ohlc

        # Now run the trade simulation
        self.simulate_trades_lean(single_trade_size=0.02)

    def simulate_trades_lean(self, single_trade_size=0.02, starting_capital=10000.0):
        """ Simulate the trades and generate the PnL curve. """
        """ single trade size is the proportion of the total equity to use for each trade."""
        """ this is the vectorized way of calculating pnl, faster but less details. Only pnl curve is available. """

        ### calculate single trade risk amount in USD(T)
        single_trade_risk = starting_capital * single_trade_size

        ### generate the empty dataframe for the PnL curve
        df_pnl_curve = pd.DataFrame(starting_capital,
                                    index=pd.date_range(start=self.bt_start, end=self.bt_end, freq=self.timeframe_low),
                                    columns=['total'])

        # df_pnl_curve = pd.DataFrame(0, index=pd.date_range(start=self.bt_start, end=self.bt_end, freq=self.timeframe_low),
        #                             columns=['cash', 'equity', 'total'])
        # df_pnl_curve.loc[self.bt_start]['cash'] = starting_capital
        # df_pnl_curve.loc[self.bt_start]['equity'] = 0


        # now loop through the master trade log df_trade_log_all to simulate each trade
        for idx, row in self.df_trade_log_all.iterrows():

            # get the symbol and OHLC data for the trade
            name_symbol = row['name_symbol']
            df_ohlc = self.dict_df_ohlc[name_symbol]

            # get the entry and exit datetime and price
            datetime_entry = row['datetime_entry']
            datetime_exit = row['datetime_exit']
            entry_price = row['entry_price']
            exit_price = row['exit_price']
            initial_risk = row['initial_risk']
            trade_direction = row['direction']

            # calculate the position size in USD(T)
            initial_risk_percent = initial_risk / entry_price
            position_size_usdt = single_trade_risk / initial_risk_percent
            if trade_direction == 'long':
                position_size_number = position_size_usdt / entry_price
            elif trade_direction == 'short':
                position_size_number = -position_size_usdt / entry_price

            # get the OHLC data for the trade
            df_ohlc_trade = df_ohlc.loc[datetime_entry:datetime_exit]

            # calculate the PnL for the trade
            # TODO - check the logic for short trades
            df_pnl_single_trade = df_ohlc_trade[['Close']] * position_size_number
            df_pnl_single_trade = df_pnl_single_trade - df_pnl_single_trade.iloc[0]['Close']
            pnl_at_exit = df_pnl_single_trade.iloc[-1]['Close']
            df_pnl_single_trade.rename(columns={'Close': 'total'}, inplace=True)
            df_pnl_single_trade.index = pd.to_datetime(df_pnl_single_trade.index)
            df_pnl_single_trade.index = df_pnl_single_trade.index.tz_convert('UTC')

            # Now the df_pnl_single_trade index is a proper DatetimeIndex with timezone awareness

            # for the pnl to be effective till the end of simulation
            new_index = pd.date_range(start=datetime_exit, end=self.bt_end, freq=self.timeframe_low)  # Adjust the frequency as needed
            df_pnl_single_trade_lasting = pd.DataFrame(pnl_at_exit, index=new_index, columns=['total'])
            df_pnl_single_trade_lasting.drop(datetime_exit, inplace=True)
            df_pnl_single_trade_merged = pd.concat([df_pnl_single_trade, df_pnl_single_trade_lasting])

            # Check for duplicate indices
            duplicate_indices = df_pnl_single_trade_merged.index.duplicated()
            has_duplicates = any(duplicate_indices)
            assert not has_duplicates, "Duplicate indices in the PnL curve!"

            # integrate this trade to the overall pnl curve
            df_pnl_curve['total'] = df_pnl_curve['total'].add(df_pnl_single_trade_merged['total'], fill_value=0)

        # save the pnl curve to class attribute
        self.df_pnl_curve = df_pnl_curve

    def visualize_pnl_curve(self):
        """Visualize the PnL curve using plotly and save as HTML."""
        import plotly.graph_objects as go
        import plotly.io as pio
        pio.renderers.default = "browser"

        # Create the figure with a line chart representing the PnL curve
        fig = go.Figure(data=[go.Scatter(x=self.df_pnl_curve.index, y=self.df_pnl_curve['total'], mode='lines')])

        # Update layout for better visibility
        fig.update_layout(
            title='PnL Curve',
            title_font_size=48,
            xaxis=dict(title='', title_font=dict(size=32)),
            yaxis=dict(title='PnL (USDT)', title_font=dict(size=32)),
            font=dict(size=32)  # This sets the global font size
        )

        return fig






if __name__ == '__main__':

    # For debugging, first set the directory paths to one level above the current directory
    dir_root = os.path.dirname(os.getcwd())
    dir_data = os.path.join(dir_root, 'module_data')
    dir_backtesting = os.path.join(dir_root, 'module_backtesting')

    pnl = PNL(job_name='chanlun20231216-234041_threehubs_1h',
              datetime_bt_start='2021-01-01 00:00:00+00:00',
              datetime_bt_end='2023-12-16 23:40:41+00:00',
              timeframe_low='1h')

    # visualize the pnl curve
    fig_pnl = pnl.visualize_pnl_curve()
    fig_pnl.show()
    fig_pnl.write_html('pnl_curve.html')
