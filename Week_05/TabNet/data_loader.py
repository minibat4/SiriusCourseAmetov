import os
import numpy as np
import pandas as pd
import pickle
import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    embedding_columns = ['workclass_cat', 'education_cat', 'marital-status_cat', 'occupation_cat',
                         'relationship_cat', 'race_cat', 'sex_cat', 'native-country_cat']
    # Removed education-num from the numeric_columns (comments are present in a dataset_utils.py)
    numeric_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

    def __init__(self, dataset_path):
        dataset_path_mod = dataset_path + '.mod.pickle'
        if not os.path.exists(dataset_path_mod):
            with open(dataset_path, 'rb') as f:
                data, self.nrof_unique_categories, self.unique_categories = pickle.load(f)

            # Checking if the data has been processed by the dataset_utils.py script
            # For each categorical column it adds a converted column with categories nums
            # Not a strong check, well said
            valuable = [1 if cat[-3:] == 'cat' else 0 for cat in data.columns]
            assert any(valuable)

            # Processing, detailed analytics is prepared in Jupyter notebook
            for i, c in enumerate(data.columns):
                data_arr = data[c].to_numpy()
                rows_if_nan = pd.isnull(data_arr)
                rows_is_nan = np.nonzero(rows_if_nan)[0]
                if len(rows_is_nan) != 0:
                    print(f'Column [{c}] NULL in #{len(rows_is_nan)} rows')
                    assert False
                    # Effective data analysis and preparation reduced processing a lot
                    # There were 2 columns:
                    # 1) education-num with numerical values, which was totally removed
                    # 2) occupation with categorical values, where <<nan>> was changed to the << ?>> category
                    # since they are semantically similar
            print('Saving modified data to', dataset_path_mod)
            with open(dataset_path_mod, 'wb') as f:
                pickle.dump([data, self.nrof_unique_categories, self.unique_categories], f)
        else:
            print('Opening modified data from', dataset_path_mod)
            with open(dataset_path_mod, 'rb') as f:
                data, self.nrof_unique_categories, self.unique_categories = pickle.load(f)

        # Change all the keys to the (key + '_cat') form
        self.nrof_unique_categories = {key + '_cat': value for key, value in self.nrof_unique_categories.items()}
        self.columns = self.embedding_columns + self.numeric_columns
        self.x = data[self.columns].reset_index(drop=True)
        self.y = np.asarray([0 if el == '<50k' else 1 for el in data['salary'].values], dtype=np.int32)

        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        row = self.x.take([idx], axis=0)

        # Do not to iterate in a loop over a dict, just loop over two fixed lists.
        # Also, do not allocate a float32 size variable for a category number since
        # it could fit in a torch.uint8 type. Further conversion for neural network layer
        # output computation is required anyway and will be performed in the neural network layers.
        x_num = torch.tensor([row[col].values for col in self.numeric_columns], dtype=torch.float32)
        x_cat = torch.tensor([row[col].values for col in self.embedding_columns], dtype=torch.uint8)

        y = np.float32(self.y[idx])

        return x_num, x_cat, y


if __name__ == '__main__':
    d = CustomDataset('../data/train_adult_cut.pickle')
    print('Dataset length', d.__len__())

    d = CustomDataset('../data/valid_adult_cut.pickle')
    print('Dataset length', d.__len__())
