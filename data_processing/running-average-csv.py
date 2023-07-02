import pandas as pd
import os

# Read the CSV file
data = pd.read_csv('../train_output.csv')

# Initialize the average column with NaN values
data['average'] = float('NaN')

# Calculate the running average
cumulative_sum = 0
for index, row in data.iterrows():
    cumulative_sum += row['total_return']
    data.at[index, 'average'] = cumulative_sum / (index + 1)

# Calculate the 50 rolling average
data['rolling_50_avg'] = data['total_return'].rolling(window=50).mean()

# Print the updated data
print(data)

# Save the updated data to a new CSV file
paths = ["C:/Users/Sam68/OneDrive/train_output_average.csv",
         "C:/Users/Sam68/OneDrive/train_output_average.xlsx",
         "C:/Users/Sam68/OneDrive/train_output_average 1.csv",
         "C:/Users/Sam68/OneDrive/train_output_average 1.xlsx"
    ]

for path in paths:
    file = path.split('/')[4]
    try:
        os.remove(path)
        print(f"removed '{file}'")
    except:
        continue


data.to_csv('C:/Users/Sam68/OneDrive/train_output_average.csv', index=False)