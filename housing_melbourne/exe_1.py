import pandas as pd

melbourne_file_path = "dataset\\melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())