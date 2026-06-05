import os
import pandas as pd

dfs = []
for d in os.listdir('./'):
    if '_out' in d:
        dfs.append(pd.read_csv(f'./{d}/perf.csv'))
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(by='Test Accuracy')
print(df)
