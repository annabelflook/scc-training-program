from os import scandir, getcwd  # listdir,
import pandas as pd


def extract_data():
    cwd = getcwd()
    training_path = cwd + r'/scc-training-program/TrainingData'

    filenames = scandir(training_path)
    dfs = [pd.read_csv(csv, index_col=0) for csv in filenames]
    CTP = ['NEC', 'Cdt', 'Cdt 1st', 'OC']
    # CTP = [name[29:33] for name in listdir(training_path)]

    arrays = [
        [[CTP[x]] * len(dfs[x].columns),
         dfs[x].columns.values]
        for x in range(len(CTP))
    ]

    tuples = [list(zip(*array)) for array in arrays]
    multituples = [pd.MultiIndex.from_tuples(tpl) for tpl in tuples]

    new_dfs = [df.set_axis(multituples[x], axis=1) for x, df in enumerate(dfs)]
    join_df = pd.concat(new_dfs, axis=1, join='inner')
    join_df['Rank'] = join_df.index.str[-19:-16].str.strip().str.replace('1st', 'Cdt 1st')
    # print(join_df)

    df = join_df.drop(['Unnamed: 1', 'Unnamed: 2'], axis=1, level=1).drop(['Person'])
    return df
