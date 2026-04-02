import pandas as pd

df = pd.read_csv('dataset_all_mias/labels/dataset_annotations_2.csv')

# print({x:i for i,x in enumerate(df['back_tissue_chr'].unique())})
# print({x:i for i,x in enumerate(df['class'].unique())})
# print({x:i for i,x in enumerate(df['severity'].unique())})

# will output something like this:
# {'G': 0, 'D': 1, 'F': 2}
# {'CIRC': 0, 'NORM': 1, 'MISC': 2, 'ASYM': 3, 'ARCH': 4, 'SPIC': 5, 'CALC': 6}
# {'B': 0, 'M': 2} <- except for this where you have to remove NaN by hand

df = df.replace({'back_tissue_chr': {'G': 0, 'D': 1, 'F': 2},
                 'class': {'CIRC': 0, 'NORM': 1, 'MISC': 2, 'ASYM': 3, 'ARCH': 4, 'SPIC': 5, 'CALC': 6},
                 'severity': {'B': 0, 'M': 1}})
df.index = df.ref_num
df = df.drop(df.columns[[0,1]], axis=1)
#TODO add *NOTE special case later
df = df['*NOTE' != df['x']]
df.to_csv('dataset_all_mias/labels/label_encoded_dataset.csv')

