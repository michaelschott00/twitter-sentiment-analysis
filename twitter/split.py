"""Split into train-, development- and testset in a reproducible way."""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

# load config
with open('configs/split.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# parameters
SEED = config['split']['seed']  # seed ensures to get the same split every time
TEST_SIZE = config['split']['test_size']
ROOT = 'data'
RAW = 'raw'
SPLITS = 'splits'

# create splits directory
if not os.path.exists(SPLITS):
    os.mkdir(SPLITS)

# train test split
df = pd.read_csv(os.path.join(ROOT, RAW, 'tweets_train.csv'))
df_train, df_dev = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, shuffle=True, stratify=df['sentiment'])

# save splits
df_train.to_csv(os.path.join(ROOT, SPLITS, "tweets_train.csv"), index=False)
df_dev.to_csv(os.path.join(ROOT, SPLITS, "tweets_dev.csv"), index=False)
