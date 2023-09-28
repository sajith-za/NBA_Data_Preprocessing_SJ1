import pandas as pd
import numpy as np
import os
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def obtain_data():
    # Checking ../Data directory presence
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'nba2k-full.csv' not in os.listdir('../Data'):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/nba2k-full.csv', 'wb').write(r.content)
        print('Loaded.')

    data_path = "../Data/nba2k-full.csv"
    return data_path


# write your code here
def columns_to_numeric(row):
    row['salary'] = pd.to_numeric(row['salary']).astype(float)
    row['height'] = pd.to_numeric(row['height'])
    row['weight'] = pd.to_numeric(row['weight'])
    return row


def clean_data(path):
    df = pd.read_csv(path)

    df['b_day'] = pd.to_datetime(df['b_day'], format='%m/%d/%y')
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
    df['team'].fillna('No Team', inplace=True)
    df['height'] = df['height'].apply(lambda x: x.split(' / ')[1])
    df['weight'] = df['weight'].apply(lambda x: x.split(' / ')[1].split(' ')[0])
    df['salary'] = df['salary'].str.replace('$', '')

    df = df.apply(columns_to_numeric, axis=1)
    df['country'] = df['country'].apply(lambda x: x if x == 'USA' else 'Not-USA')
    df['draft_round'] = df['draft_round'].str.replace('Undrafted', '0')

    return df


def feature_data(data):
    df = data
    # Setting the version date formats
    df['version'] = pd.to_datetime(df['version'].
                                   apply(lambda x: '2020' if x == 'NBA2k20' else '2021'), format='%Y')

    df['age'] = ((df['version'] - df['b_day']) / np.timedelta64(1, 'Y')). \
        apply(np.ceil).astype(int)

    df['experience'] = ((df['version'] - df['draft_year']) / np.timedelta64(1, 'Y')). \
        apply(np.floor).astype(int)

    df['bmi'] = df['weight'] / df['height'] ** 2
    df.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'], inplace=True)

    '''Removing High Cardinality'''
    categorical_columns = [col for col in df.columns if (df[col].dtype == object)]
    columns_with_high_cardinality = [col for col in categorical_columns if df[col].nunique() > 50]
    df.drop(columns_with_high_cardinality, axis=1, inplace=True)

    return df


def multicol_data(data):
    df = data
    corr = df._get_numeric_data().corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))\

    col_keep = [column for column in upper.columns if (any(upper[column] > 0.5) or any(upper[column] < -0.5))]
    row_keep = [row for row in upper.index if (any(upper.loc[row] > 0.5) or any(upper.loc[row] < -0.5))]
    corr_cols = list(set(col_keep).union(set(row_keep)))

    min_corr = 100.0
    ref = ''

    for i in corr_cols:
        temp = df[i].corr(data['salary'])

        if temp < min_corr:
            min_corr = temp
            ref = i

    df.drop(ref, axis=1, inplace=True)

    return df


def transform_data(data):
    y = data['salary']

    # Setting up the numerical scalar
    df_num = data.select_dtypes('number').drop('salary', axis=1)

    scaler_std = StandardScaler()
    df_standard = scaler_std.fit_transform(df_num)
    df_standard = pd.DataFrame(df_standard, columns=df_num.columns)

    df_cat = data.select_dtypes('object')
    hot_enc = OneHotEncoder()
    df_onehotenc = hot_enc.fit_transform(df_cat).toarray()

    gr = hot_enc.categories_
    list_of_columns = np.concatenate(gr).tolist()

    df_onehotenc = pd.DataFrame(df_onehotenc, columns=list_of_columns)
    x = pd.concat([df_standard, df_onehotenc], axis=1)
    return x, y


def main():
    path_name = obtain_data()
    df_cleaned = clean_data(path_name)
    df_featured = feature_data(df_cleaned)
    df = multicol_data(df_featured)
    X, y = transform_data(df)


if __name__ == "__main__":
    main()
