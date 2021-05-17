import main
from sklearn import preprocessing


class Rank:
    def __init__(self, *ranks, syllabus):
        self.ranks = [rank for rank in ranks]
        print(self.ranks)
        self.syllabus = syllabus + ['Rank']

    def find_best_modules(self, merged_df):
        """Find the best modules for these ranks"""
        filtered_df = main.filter_data(merged_df, self.ranks)
        date_matrix = main.create_date_matrix(filtered_df)

        minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 10))

        cdt_vector = main.create_cdt_vector(date_matrix)
        scaled_cdt_vector = minmax_scaler.fit_transform(cdt_vector.T)  # 20 x 1 vector

        mod_vector = main.create_module_vector(date_matrix)
        scaled_mod_vector = minmax_scaler.fit_transform(mod_vector.T)  # 48 x 1 vector

        ratings = main.create_rating_matrix(cdt_vec=scaled_cdt_vector, mod_vec=scaled_mod_vector, date_mat=date_matrix)

        best_modules = ratings.sum(axis=0)
        best_modules_sorted = best_modules.replace(0, (best_modules.max() + 1)).sort_values(ascending=False)

        return best_modules_sorted

    def show_best_modules(self, merged_df, n=5):

        merged_df = merged_df.loc[:, self.syllabus]
        best_modules_sorted = self.find_best_modules(merged_df)

        print(best_modules_sorted.head(n))

        return best_modules_sorted


if __name__ == '__main__':
    df = main.extract_data()
    cdt1 = Rank('Cdt 1st', syllabus=['Cdt 1st'])
    cdt = Rank('Cdt', syllabus=['Cdt'])
    mixed = Rank('Cdt 1st', 'OC', syllabus=['Cdt 1st', 'OC'])
    cdt.show_best_modules(df, n=10)
    cdt1.show_best_modules(df, n=10)
    mixed.show_best_modules(df, n=5)
