from binance.client import Client
import pandas as pd
import datetime

api_key = 'vlNCynmULdR2rgKD3rkSwKPGurX80tLirjhrCkQRIjrqacFYefsMSV7TeRkLLBzO'
api_secret = 'Kk7NjgfFVCGbwYy0vpwMFE4eXmUAGEdhdnKwpl3uBB704r4fqy548hP1SOYn9boc'

client = Client(api_key, api_secret)

# Get 5 minute klines from 01/01/2021 to 01/01/2023
start_date = '1 May 2023'
end_date = '1 Jun 2023'

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, start_date, end_date)

# Create a dataframe
df = pd.DataFrame(klines, columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])

# Convert timestamp to date
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')

# Select only the required columns
df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

# Rename 'open_time' to 'Date'
df.rename(columns={'open_time': 'Date', 
                   'open': 'Open', 
                   'high': 'High',
                   'low': 'Low',
                   'close': 'Close',
                   'volume': 'Volume'}, inplace=True)

# Save to csv
df.to_csv('BTC_5min_1mon_data.csv', index=False)
