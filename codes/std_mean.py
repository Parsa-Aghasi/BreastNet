from dataset_build import mias

dataset = mias('dataset_all_mias/labels/label_encoded_dataset.csv', 
               'dataset_all_mias/dataset_jpeg')

mean = 0.
std = 0.
nb_samples = len(dataset)

for data in dataset:
    data = data[0]
    mean += data.mean()
    std += data.std()

mean /= nb_samples
std /= nb_samples
print(f'mean: {mean}, std: {std}') 
# mean: 54.35747528076172, std: 71.1281967163086