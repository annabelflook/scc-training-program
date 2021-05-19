import pandas as pd
import numpy as np
from sklearn import preprocessing
from ExtractData import extract_data



def filter_data(merged_df, ranks: list, syllabus='all'):
    """Filters data by rank, refactors data for use, returns a filtered pd.DataFrame"""
    filtered_data = merged_df[merged_df['Rank'].isin(ranks)]
    fd1 = filtered_data.drop(['Rank'], axis=1)

    if syllabus != 'all':
        fd1 = fd1[syllabus]

    return fd1


def create_date_matrix(filtered_data):
    datetime_df = filtered_data.apply(pd.to_datetime, errors='coerce', dayfirst=True)
    numeric_data = datetime_df.apply(pd.to_numeric)

    # Scaled empirically
    scaled_matrix = ((100 / (numeric_data.apply(np.log) - 41)) - 110)
    scaled_matrix.fillna(0, inplace=True)

    assert scaled_matrix.shape == (len(scaled_matrix.index), len(scaled_matrix.columns))

    return scaled_matrix


def create_cdt_vector(date_mat):
    """Create 1 x len(date_mat.index) vector, as the summed dates for each pin number"""
    cdt_df = date_mat.apply(sum, axis=1)
    cdt_vec = np.array([cdt_df])

    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 10))
    scaled_cdt_vector = minmax_scaler.fit_transform(cdt_vec.T)  # 20 x 1 vector

    assert cdt_vec.shape == (1, len(date_mat.index))
    assert scaled_cdt_vector.shape == (len(date_mat.index), 1)

    return scaled_cdt_vector


def create_module_vector(date_mat):
    """Create 1 x 48 vector, as the summed dates for each pin number"""
    mod_df = 1 / date_mat.apply(sum, axis=0)
    mod_df = mod_df.replace(np.inf, 0)
    assert sum(mod_df) < np.inf

    mod_vec = np.array([mod_df])

    minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 10))
    scaled_mod_vector = minmax_scaler.fit_transform(mod_vec.T)

    assert mod_vec.shape == (1, len(date_mat.columns))
    assert scaled_mod_vector.shape == (len(date_mat.columns), 1)

    return scaled_mod_vector


def create_rating_matrix(cdt_vec, mod_vec, date_mat):

    assert cdt_vec.shape == (len(date_mat.index), 1)
    assert mod_vec.shape == (len(date_mat.columns), 1)

    cdt_mod_mat = np.matmul(cdt_vec, mod_vec.T)  # 20 x 48
    assert cdt_mod_mat.shape == (len(date_mat.index), len(date_mat.columns))

    rating_mat = date_mat * cdt_mod_mat
    return rating_mat


if __name__ == '__main__':
    cdt_csv = extract_data()

    rank_list = ['Cdt']
    filtered_df = filter_data(cdt_csv, rank_list)

    date_matrix = create_date_matrix(filtered_df)
    cdt_vector = create_cdt_vector(date_matrix)
    mod_vector = create_module_vector(date_matrix)

    ratings = create_rating_matrix(cdt_vec=cdt_vector, mod_vec=mod_vector, date_mat=date_matrix)

    best_modules_sorted = ratings.sum(axis=0).sort_values(ascending=False)
    print(best_modules_sorted.head(4))

