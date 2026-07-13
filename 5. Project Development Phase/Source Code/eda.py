import pandas as pd

df = pd.read_csv("hdi_dataset.csv")

for col in df.columns:
    if "Expected Years of Schooling" in col:
        print(col)