# Import those libraries
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import collections
from sklearn.decomposition import PCA
# create 2D PCA space from original data (X)


def correlation_analysis(df, correlation_axis="Overall satisfaction"):
    correlation = {}
    overall = df[correlation_axis]
    for col in df.columns:
        feature = df[col]
        corr, _ = pearsonr(feature, overall)
        if col not in correlation:
            correlation[col] = corr
        # print(col, ":", corr)

    sorted_dict = collections.OrderedDict(correlation)
    # print(sorted_dict)
    for key in sorted_dict:
        print(key, ":", sorted_dict[key])


def pca_feature_analysis(df):
    data_x = df.iloc[:, :-1].values
    data_y = df.iloc[:, -1].values
    pca = PCA(5)
    X_pca = pca.fit_transform(data_x)
    print(pca.explained_variance_ratio_)
    print((pca.components_))
    for component in pca.components_:
        index = sorted(range(len(component)), key=lambda i: component[i])[-5:]
        print("\ncomponent .....\n")
        for i in index:
            print(df.columns[i])

    fet =  np.dot(data_x.T, X_pca)
    print(fet.shape)
    plt.imshow(fet, cmap='hot', interpolation='nearest')
    # plt.colorbar()
    plt.show()




if __name__ == '__main__':
    df = pd.read_csv("airport.csv")
    # correlation_analysis(df)
    pca_feature_analysis(df)

