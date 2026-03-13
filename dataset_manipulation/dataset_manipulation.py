import pandas as pd

df = pd.read_csv('labels/dataset_annotations.csv')
df.index = df.ref_num
df = df.drop(df.columns[[0,1]], axis=1)
df.to_csv('labels/dataset_annotations_2.csv')