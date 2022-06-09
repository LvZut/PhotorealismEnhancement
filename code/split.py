from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


csv_file = 'hddCARLA_files.csv'
df = pd.read_csv('data/hddCARLA/hddCARLA_files.csv', sep=',')
train, val, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])
train.to_csv('data/hddCARLA/train.txt', index=False)
val.to_csv('data/hddCARLA/val.txt',index=False)
test.to_csv('data/hddCARLA/test.txt', index=False)
