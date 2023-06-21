import requests
import csv
import time
from datetime import datetime, timedelta

# Define the API URL
url = "https://min-api.cryptocompare.com/data/v2/histominute"

# Define the parameters
params = {
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000  # Max limit per request
}

# Prepare the CSV file
with open('bitcoin_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])

    # Loop over each day in the past week
    for i in range(7):
        # Calculate the end timestamp for this day
        end_time = datetime.now() - timedelta(days=i)
        params["toTs"] = int(end_time.timestamp())

        # Send the request
        response = requests.get(url, params=params)

        # Parse the response
        data = response.json()['Data']['Data']

        # Write each data point
        for point in data:
            date = datetime.fromtimestamp(point['time']).strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([date, point['open'], point['high'], point['low'], point['close'], point['volumeto']])