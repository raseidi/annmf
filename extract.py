import os
import time
from annmf import utils, estimators
import numpy as np
from pymfe.mfe import MFE

import argparse


def read_args(add_help=True):
    """Get args parser"""
    parser = argparse.ArgumentParser(
        description="Aproximate nearest neighbor meta-features.",
        add_help=add_help,
    )

    parser.add_argument(
        "--dataset-path",
        default="data/base2.txt",
        type=str,
        help="Dataset directory path.",
    )
    
    parser.add_argument(
        "--nr-inst",
        default=None,
        type=int,
        help="Number of instances to use to extract meta-features.",
    )
    
    # # optional meta-features
    # parser.add_argument(
    #     "--generic",
    #     default=True,
    #     type=bool,
    #     help="Extract generic meta-features.",
    # )

    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()
    file_name = os.path.basename(args.dataset_path).split(".")[0]
    
    mf = utils.read_metafeatures()
    print("="*80)
    print(f"[+] Extracting MFs from {file_name}")
    print("="*80)

    # read the dataset
    X = utils.read_data(args.dataset_path, as_dataframe=True)

    # number of instances to extract meta-features; it must be less or equal than the dataset size
    if args.nr_inst is None:
        nr_inst = len(X)
    else:
        nr_inst = args.nr_inst

    start = time.time()
    try:
        data = X.sample(nr_inst).values
    except:
        data = X.sample(len(X)).values

    dist = utils.get_distances(data)

    # id-based measures
    print(f"\tANN meta-features")
    lids, (hist, _) = estimators.get_lids(dist)
    n_pcs = estimators.get_pcs(data)

    # concentration-based measures
    rc = estimators.get_rc(dist)
    rv = estimators.get_rv(data)

    # general measures
    print(f"\tGeneric meta-features")
    mfe = MFE(
        groups=["general", "statistical", "info-theory"], suppress_warnings=True
    )
    mfe.fit(data, suppress_warnings=True)
    ft = mfe.extract(suppress_warnings=True)

    # formating output
    mf = utils.format_output(
        mf,
        lids=lids,
        n_pcs=n_pcs,
        hist=hist,
        ft=ft,
        rc=rc,
        rv=rv,
        file_name=file_name,
    )
    mf.loc[file_name, "elapsed_time(secs)"] = time.time() - start
    mf.to_csv("data/metafeatures.csv", index=True)
