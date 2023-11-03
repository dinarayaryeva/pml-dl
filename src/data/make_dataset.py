import re
import numpy as np
import pandas as pd
import zipfile
import sys


def to_lowercase(text):
    """Convert all characters to lowercase in string of text"""
    return text.lower()


def remove_punctuation(text):
    """Remove punctuation from string of text"""
    regex = re.compile('[' + ',' + '0-9\r\t\n]')
    nopunct = regex.sub(" ", text)
    return nopunct


def preprocess(sample):

    sample = sample.lower()
    sample = remove_punctuation(sample)
    sample = " ".join(sample.split())

    return sample


def pipe(x):
    return preprocess(x)


def preprocess_dataset(df):

    df['source'] = df['source'].apply(pipe)
    df['target'] = df['target'].apply(pipe)

    return df


path = sys.path[1]

with zipfile.ZipFile(path + f'/data/raw/filtered_paranmt.zip', 'r') as zip_ref:
    zip_ref.extractall(path + f'/data/raw')

df = pd.read_table(path + '/data/raw/filtered.tsv', index_col=0)
df.reset_index(drop=True, inplace=True)
df.head()

toxic = []
neutral = []
for i, row in df.iterrows():
    if row.ref_tox > row.trn_tox:
        toxic.append(row.reference)
        neutral.append(row.translation)
    else:
        neutral.append(row.reference)
        toxic.append(row.translation)

dataset = pd.DataFrame({'source': toxic, 'target': neutral})

small_df = dataset.sample(frac=0.1, random_state=42)

train, validate, test = np.split(small_df.sample(frac=1, random_state=42), [
                                 int(.6*len(small_df)), int(.8*len(small_df))])

train = preprocess_dataset(train)
validate = preprocess_dataset(validate)
test = preprocess_dataset(test)

train.to_csv(path + "data/interim/1/train.csv")
validate.to_csv(path + "data/interim/1/validate.csv")
test.to_csv(path + "data/interim/1/test.csv")
