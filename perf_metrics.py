import os
import argparse
import pandas as pd

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./', required=False)
args = parser.parse_args()

dfs = []
for d in os.listdir(args.dir):
    if '_out' in d:
        try:
            dfs.append(pd.read_csv(f'{args.dir}/{d}/perf.csv'))
        except (FileNotFoundError, NotADirectoryError):
            pass
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(by='Test Accuracy')
print(df)
