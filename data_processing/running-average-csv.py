import pandas as pd

# Read the CSV file
data = pd.read_csv('train_output.csv')

# Initialize the average column with NaN values
data['average'] = float('NaN')

# Calculate the running average
cumulative_sum = 0
for index, row in data.iterrows():
    cumulative_sum += row['total_return']
    data.at[index, 'average'] = cumulative_sum / (index + 1)

# Print the updated data
print(data)

# Save the updated data to a new CSV file
data.to_csv('train_output_average.csv', index=False)