import pandas as pd

# Replace 'your_data.csv' with the path to your CSV file
file_path = r'D:\I4-internship\Internship\data\train_data.csv'

# Read the CSV file
data = pd.read_csv(file_path)

# Count the number of rows

row_count = len(data)

print(f"Number of rows in the CSV file: {row_count}")
