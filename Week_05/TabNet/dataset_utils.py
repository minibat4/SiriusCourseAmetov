import pickle
from pprint import pprint

import numpy as np
import pandas as pd

embedding_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race',
                     'sex', 'native-country']

nrof_unique_categories = {}
unique_categories = {}

with open('../data/train_adult.pickle', 'rb') as f:
    train_data, _, _ = pickle.load(f)
with open('../data/valid_adult.pickle', 'rb') as f:
    val_data, _, _ = pickle.load(f)

# region Important novelty
# There are education and education-num columns in the table.
# They totally correspond to each other but the education-num column has nan values.
# It would be convenient to remove one.
# Nevertheless, it would be used with embeddings.
# Since the numerical categorical column is easy to mislabel with a simple numerical column,
# it would be safer to delete <<education-num>> and leave <<education>>

# Checking if its a raw table
assert 'education-num' in train_data.columns
assert 'education-num' in val_data.columns

data_education = train_data['education'].to_numpy()
data_education_num = train_data['education-num'].to_numpy()

n_nan_data_education = sum(pd.isnull(data_education))
n_nan_data_education_num = sum(pd.isnull(data_education_num))

print('Rows with nan values')
print('<<education>>    ', n_nan_data_education)
print('<<education-nan>>', n_nan_data_education_num)

# There is a full correspondence of string and numerical values
rows_if_nan = pd.isnull(data_education_num)
rows_is_nan = np.nonzero(rows_if_nan)[0]
rows_is_not_nan = np.nonzero(1 - rows_if_nan)[0]
s = set(list(zip(data_education[rows_is_not_nan], data_education_num[rows_is_not_nan])))
print('Set of pairs, total amount of pairs:', len(s))
pprint(s)
d = {key: value for (key, value) in s}
print('Seems there is a direct mapping')
pprint(d)
# Here we could map the nan values in the <<education-num>> but
print('Deleting <<education-num>> because building an embedding for the <<education>>')
del train_data['education-num']
# Modify a valid data the same way as train data
# Hope there are no reasons to perform an analysis on a combined dataset
del val_data['education-num']
# endregion

print('*' * 50)
print('Going in a loop over the whole embedding columns')

for cat in embedding_columns:
    str_data_train = train_data[cat].to_numpy(dtype=str)
    str_data_val = val_data[cat].to_numpy(dtype=str)
    # Get common data
    unique_train = np.unique(str_data_train)
    unique_val = np.unique(str_data_val)
    unique = set(unique_train) | set(unique_val)
    unique = np.array(list(unique), dtype=str)
    # Join <<nan>> and << ?>> embedding categories to reduce the task
    # complexity and remove excessive computation.
    # Also <<nan>> and << ?>> are almost equal in their meaning.
    if 'nan' in unique:
        train_data_indexes = np.where(str_data_train == 'nan')[0]
        val_data_indexes = np.where(str_data_val == 'nan')[0]
        print(f'Warning: detected nan! '
              f'{len(train_data_indexes)} in train & '
              f'{len(val_data_indexes)} in val')
        # Remove the nan category
        unique_index = np.where(unique == 'nan')[0][0]
        if ' ?' not in unique:
            unique[unique_index] = ' ?'
        else:
            l_unique = list(unique)
            del l_unique[unique_index]
            unique = np.array(l_unique, dtype=str)
        # Change in train and valid data
        cat_n = np.where(train_data.columns == cat)[0][0]
        train_data.iloc[train_data_indexes, cat_n] = ' ?'
        cat_n = np.where(val_data.columns == cat)[0][0]
        val_data.iloc[val_data_indexes, cat_n] = ' ?'
        # Safety check
        str_data_train = train_data[cat].to_numpy(dtype=str)
        str_data_val = val_data[cat].to_numpy(dtype=str)
        train_data_indexes = np.where(str_data_train == 'nan')[0]
        assert len(train_data_indexes) == 0
        val_data_indexes = np.where(str_data_val == 'nan')[0]
        assert len(val_data_indexes) == 0

    unique_categories[cat] = unique
    nrof_unique_categories[cat] = len(unique)
    print(f'Category [{cat:20}] Unique values, total [{len(unique):5}]')
    pprint(np.array(list(zip(np.arange(len(unique)), unique))))
    # Create a dict once and use it many-many times when changing the str_data
    d_unique = {value: i for i, value in enumerate(unique)}
    # Modify train and val data
    train_data[cat + '_cat'] = [d_unique[value] for value in str_data_train]
    val_data[cat + '_cat'] = [d_unique[value] for value in str_data_val]

with open('../data/train_adult_cut.pickle', 'wb') as f:
    pickle.dump([train_data, nrof_unique_categories, unique_categories], f)
    del train_data

with open('../data/valid_adult_cut.pickle', 'wb') as f:
    pickle.dump([val_data, nrof_unique_categories, unique_categories], f)
    del val_data
