from main import *

cdt_csv = pd.read_csv(r'C:\Users\AnnabelFlook\PycharmProjects\CadetsTP\cadet_training_program_matrix.csv')


class Rank:
    def __init__(self, *ranks):
        self.ranks = [rank for rank in ranks]

    def find_best_modules(self, merged_df):
        """Find the best modules for these ranks"""
        filtered_df = filter_data(merged_df, self.ranks)
        date_matrix = create_date_matrix(filtered_df)

        minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 10))

        cdt_vector = create_cdt_vector(date_matrix)
        scaled_cdt_vector = minmax_scaler.fit_transform(cdt_vector.T)  # 20 x 1 vector
        assert cdt_vector.shape == (1, len(date_matrix.index))
        assert scaled_cdt_vector.shape == (len(date_matrix.index), 1)

        mod_vector = create_module_vector(date_matrix)
        scaled_mod_vector = minmax_scaler.fit_transform(mod_vector.T)  # 48 x 1 vector
        assert mod_vector.shape == (1, 48)
        assert scaled_mod_vector.shape == (48, 1)

        ratings = create_rating_matrix(cdt_vec=scaled_cdt_vector, mod_vec=scaled_mod_vector, date_mat=date_matrix)

        best_modules_sorted = ratings.sum(axis=0).sort_values(ascending=False)

        return best_modules_sorted

    def show_best_modules(self, merged_df, n=5):
        best_modules_sorted = self.find_best_modules(merged_df)
        print(best_modules_sorted.head())

        return None


print('__name__')
cdt1 = Rank('Cdt 1st')
cdt = Rank('Cdt')
cdt.show_best_modules(cdt_csv)