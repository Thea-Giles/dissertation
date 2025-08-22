import pandas as pd
import math

# Path to the original CSV file
input_file = r"C:\Users\noah\thea\dissertation\src\data_collection\data\combined_cleaned_tweets2.csv"

# Read the CSV file
print(f"Reading {input_file}...")
df = pd.read_csv(input_file)

# Calculate the number of rows per split
total_rows = len(df)
rows_per_split = math.ceil(total_rows / 8)
print(f"Total rows: {total_rows}")
print(f"Rows per split: {rows_per_split}")

# Split the dataframe and save each part
for i in range(8):
    start_idx = i * rows_per_split
    end_idx = min((i + 1) * rows_per_split, total_rows)
    
    # Create a slice of the dataframe
    df_slice = df.iloc[start_idx:end_idx]
    
    # Define the output file path
    output_file = f"C:\\Users\\noah\\thea\\dissertation\\src\\data_collection\\data\\combined_cleaned_tweets2_part{i+1}.csv"
    
    # Save the slice to a CSV file
    print(f"Saving part {i+1} ({len(df_slice)} rows) to {output_file}...")
    df_slice.to_csv(output_file, index=False)

print("Splitting completed successfully!")