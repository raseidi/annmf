import numpy as np
import pandas as pd

def read_data(path, as_dataframe=False):
    data = pd.read_csv(path, sep=' ', header=None)
    data.drop_duplicates(inplace=True)
    data.dropna(axis=1, inplace=True)

    if as_dataframe:
        return data
    else:
        return data.values

def get_distances(X, k=100, n_queries=10000, shuffle=True, return_ix=False):
    '''
        Returns distances to the `n_queries` selected.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_features)
            The dataset.
        
        k : int, default=100
            Number of neighbors of each element.

        n_queries : int, default=10000
            Number of queries to compute.
            We use (n_samples - n_queries) to build the graph,
            and n_queries to compute the distances.
        
        shuffle : bool, default=True
            If True, it shuffles the dataset before selecting
            the query elements.

        return_ix : bool, default=False
            Returns indices if True.   

        Returns
        -------
        dist : array-like, shape(n_queries, n_neighbors)
            Array considering the `n_neighbors` closest points
            to each element.
        ind : array-like, shape(n_queries, n_neighbors)
            Indices of the nearest points in the distance matrix,
            only if return_ix=True
    '''
    from sklearn.neighbors import NearestNeighbors
    if shuffle:
        np.random.shuffle(X)

    if X.shape[0] <= n_queries:
        n_queries = 100

    X_train = X[n_queries:, :]
    X_test = X[:n_queries, :]

    neighb = NearestNeighbors(n_neighbors=k+1, n_jobs=-1, algorithm='ball_tree').fit(X_train)
    dist, ind = neighb.kneighbors(X_test)

    if return_ix:
        return dist, ind
    else:
        return dist

