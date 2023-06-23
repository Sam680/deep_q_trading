import pandas as pd

# Read the CSV file
df = pd.read_csv('BTCUSDT-data.csv')

# Calculate percentage difference for each column
columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
    'Bid1_Price', 'Bid1_Quantity', 'Ask1_Price', 'Ask1_Quantity', 
    'Bid2_Price', 'Bid2_Quantity', 'Ask2_Price', 'Ask2_Quantity'
]

for column in columns:
    df[column + '_Delta'] = (df[column] - df[column].shift(1)) / df[column].shift(1) * 100

# Drop the columns that aren't the calculated percentages
df = df.drop( columns=[ 'index', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Bid1_Price', 'Bid1_Quantity', 'Ask1_Price', 'Ask1_Quantity',
    'Bid2_Price', 'Bid2_Quantity', 'Ask2_Price', 'Ask2_Quantity',
    'Bid3_Price', 'Bid3_Quantity', 'Ask3_Price' , 'Ask3_Quantity',
    'Bid4_Price', 'Bid4_Quantity', 'Ask4_Price', 'Ask4_Quantity',
    'Bid5_Price', 'Bid5_Quantity', 'Ask5_Price', 'Ask5_Quantity'
])
df = df.dropna()

# Scale the data using MinMaxScaler
# scaler = MinMaxScaler()
# scaled_columns = [column + '_Delta' for column in columns]
# df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

# Print the resulting data frame
print(df)

# Write the processed data frame to a CSV file
df.to_csv('delta-data.csv', index=False)

print("Processed data has been written to delta-data.csv")