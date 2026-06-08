import os
import pandas as pd

dfs = []
for d in os.listdir('./'):
    if '_out' in d:
        try:
            dfs.append(pd.read_csv(f'./{d}/perf.csv'))
        except (FileNotFoundError, NotADirectoryError):
            pass
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(by='Test Accuracy')
print(df)
