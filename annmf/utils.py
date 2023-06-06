import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy


def read_data(path, as_dataframe=False):
    data = pd.read_csv(path)
    data.drop_duplicates(inplace=True)
    data.dropna(axis=1, inplace=True)

    if as_dataframe:
        return data
    else:
        return data.values


def get_distances(X, k=100, n_queries=10000, shuffle=True, return_ix=False):
    """
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
    """
    from sklearn.neighbors import NearestNeighbors

    if shuffle:
        np.random.shuffle(X)

    if X.shape[0] <= n_queries:
        n_queries = 100

    X_train = X[n_queries:, :]
    X_test = X[:n_queries, :]

    neighb = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1, algorithm="ball_tree").fit(
        X_train
    )
    dist, ind = neighb.kneighbors(X_test)

    if return_ix:
        return dist, ind
    else:
        return dist

def read_metafeatures():
    import os
    if not os.path.exists("data/"):
        # create data folder
        os.makedirs("data/")
    
    try:
        mf = pd.read_csv("data/metafeatures.csv", index_col=0)
    except:
        mf = None
    return mf

def format_output(mf, **args):
    lids = args["lids"]
    n_pcs = args["n_pcs"]
    ft = args["ft"]
    hist = args["hist"]
    file_name = args["file_name"]
    rc = args["rc"]
    rv = args["rv"]

    # formating output dataframe
    if mf is None:
        mf = pd.DataFrame(
            data=np.array(ft[1]).reshape(1, -1), columns=ft[0], index=[file_name]
        )
    else:
        new = pd.DataFrame(
            data=np.array(ft[1]).reshape(1, -1), columns=ft[0], index=[file_name]
        )
        mf = pd.concat([mf, new])

    mf.loc[file_name, "n_pcs"] = n_pcs
    mf.loc[file_name, "rv"] = rv
    mf.loc[file_name, "lid_mean"] = np.mean(lids)
    mf.loc[file_name, "lid_median"] = np.median(lids)
    mf.loc[file_name, "lid_std"] = np.std(lids)
    mf.loc[file_name, "lid_kurtosis"] = kurtosis(lids)
    mf.loc[file_name, "lid_skew"] = skew(lids)
    mf.loc[file_name, "lid_entropy"] = entropy(lids)
    mf.loc[file_name, "lid_hist0"] = hist[0] / len(lids)
    mf.loc[file_name, "lid_hist1"] = hist[1] / len(lids)
    mf.loc[file_name, "lid_hist2"] = hist[2] / len(lids)
    mf.loc[file_name, "lid_hist3"] = hist[3] / len(lids)
    mf.loc[file_name, "lid_hist4"] = hist[4] / len(lids)
    mf.loc[file_name, "lid_hist5"] = hist[5] / len(lids)
    mf.loc[file_name, "lid_hist6"] = hist[6] / len(lids)
    mf.loc[file_name, "lid_hist7"] = hist[7] / len(lids)
    mf.loc[file_name, "lid_hist8"] = hist[8] / len(lids)
    mf.loc[file_name, "lid_hist9"] = hist[9] / len(lids)

    hist, _ = np.histogram(rc, 10)
    mf.loc[file_name, "rc_mean"] = np.mean(rc)
    mf.loc[file_name, "rc_median"] = np.median(rc)
    mf.loc[file_name, "rc_std"] = np.std(rc)
    mf.loc[file_name, "rc_kurtosis"] = kurtosis(rc)
    mf.loc[file_name, "rc_skew"] = skew(rc)
    mf.loc[file_name, "rc_entropy"] = entropy(rc)
    mf.loc[file_name, "rc_hist0"] = hist[0] / len(rc)
    mf.loc[file_name, "rc_hist1"] = hist[1] / len(rc)
    mf.loc[file_name, "rc_hist2"] = hist[2] / len(rc)
    mf.loc[file_name, "rc_hist3"] = hist[3] / len(rc)
    mf.loc[file_name, "rc_hist4"] = hist[4] / len(rc)
    mf.loc[file_name, "rc_hist5"] = hist[5] / len(rc)
    mf.loc[file_name, "rc_hist6"] = hist[6] / len(rc)
    mf.loc[file_name, "rc_hist7"] = hist[7] / len(rc)
    mf.loc[file_name, "rc_hist8"] = hist[8] / len(rc)
    mf.loc[file_name, "rc_hist9"] = hist[9] / len(rc)

    # mf = mf.astype(float)
    return mf

