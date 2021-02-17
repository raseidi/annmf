import os
import time
import utils
import estimators
import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from scipy.stats import kurtosis, skew, entropy

# os.chdir('../')

def format_output(mf, **args):
    lids = args['lids']
    n_pcs = args['n_pcs']
    ft = args['ft']
    hist = args['hist']
    file_name = args['file_name']
    rc = args['rc']
    rv = args['rv']

    # formating output dataframe
    if mf is None:
        mf = pd.DataFrame(data=np.array(ft[1]).reshape(1, -1), columns=ft[0], index=[file_name])
    else:
        new = pd.DataFrame(data=np.array(ft[1]).reshape(1, -1), columns=ft[0], index=[file_name])
        mf = pd.concat([mf, new])

    mf.loc[file_name, 'n_pcs'] = n_pcs
    mf.loc[file_name, 'rv'] = rv
    mf.loc[file_name, 'lid_mean'] = np.mean(lids)
    mf.loc[file_name, 'lid_median'] = np.median(lids)
    mf.loc[file_name, 'lid_std'] = np.std(lids)
    mf.loc[file_name, 'lid_kurtosis'] = kurtosis(lids)
    mf.loc[file_name, 'lid_skew'] = skew(lids)
    mf.loc[file_name, 'lid_entropy'] = entropy(lids)
    mf.loc[file_name, 'lid_hist0'] = hist[0] / len(lids)
    mf.loc[file_name, 'lid_hist1'] = hist[1] / len(lids)
    mf.loc[file_name, 'lid_hist2'] = hist[2] / len(lids)
    mf.loc[file_name, 'lid_hist3'] = hist[3] / len(lids)
    mf.loc[file_name, 'lid_hist4'] = hist[4] / len(lids)
    mf.loc[file_name, 'lid_hist5'] = hist[5] / len(lids)
    mf.loc[file_name, 'lid_hist6'] = hist[6] / len(lids)
    mf.loc[file_name, 'lid_hist7'] = hist[7] / len(lids)
    mf.loc[file_name, 'lid_hist8'] = hist[8] / len(lids)
    mf.loc[file_name, 'lid_hist9'] = hist[9] / len(lids)

    hist, _ = np.histogram(rc, 10)
    mf.loc[file_name, 'rc_mean'] = np.mean(rc)
    mf.loc[file_name, 'rc_median'] = np.median(rc)
    mf.loc[file_name, 'rc_std'] = np.std(rc)
    mf.loc[file_name, 'rc_kurtosis'] = kurtosis(rc)
    mf.loc[file_name, 'rc_skew'] = skew(rc)
    mf.loc[file_name, 'rc_entropy'] = entropy(rc)
    mf.loc[file_name, 'rc_hist0'] = hist[0] / len(rc)
    mf.loc[file_name, 'rc_hist1'] = hist[1] / len(rc)
    mf.loc[file_name, 'rc_hist2'] = hist[2] / len(rc)
    mf.loc[file_name, 'rc_hist3'] = hist[3] / len(rc)
    mf.loc[file_name, 'rc_hist4'] = hist[4] / len(rc)
    mf.loc[file_name, 'rc_hist5'] = hist[5] / len(rc)
    mf.loc[file_name, 'rc_hist6'] = hist[6] / len(rc)
    mf.loc[file_name, 'rc_hist7'] = hist[7] / len(rc)
    mf.loc[file_name, 'rc_hist8'] = hist[8] / len(rc)
    mf.loc[file_name, 'rc_hist9'] = hist[9] / len(rc)

    # mf = mf.astype(float)
    return mf

if __name__ == '__main__':
    #curr_dir = '/media/hd/datasets/syn/'
    curr_dir = '/media/hd/datasets/test/'
    files = [os.path.join(curr_dir, f) for f in os.listdir(curr_dir) if os.path.isfile(os.path.join(curr_dir, f))]
    info_bases = pd.read_csv('data/info_bases.csv')

    try:
        mf = pd.read_csv('data/metafeatures.csv', index_col=0)
    except:
        mf = None

    for f in files:
        file_name = f.split('/')[-1].split('.')[0]
        if(file_name.startswith('gist')):
            continue
        print(f'-------------- Extracting MFs from {file_name} --------------')

        X = utils.read_data(f, as_dataframe=True)

        if file_name in info_bases.base.unique():
            nr_inst_values = info_bases[info_bases.base == file_name].nr_inst.unique()
        else:
            nr_inst_values = [len(X)]

        for nr in nr_inst_values:
            start = time.time()

            try:
                data = X.sample(nr).values
            except:
                data = X.sample(len(X)).values

            dist = utils.get_distances(data)

            # id-based measures
            print(f'\t[{file_name}_{nr}] Extracting ID-based measures')
            lids = estimators.get_lids(dist)
            hist, _ = np.histogram(lids, 10)
            n_pcs = estimators.get_pcs(data)

            # concentration-based measures
            rc = estimators.get_rc(dist)
            rv = estimators.get_rv(data)

            # general measures
            print(f'\t[{file_name}_{nr}] Extracting generic meta-features')
            mfe = MFE(groups=["general", "statistical", "info-theory"], suppress_warnings=True)
            mfe.fit(data, suppress_warnings=True)
            ft = mfe.extract(suppress_warnings=True)

            # formating output
            file_name_ = file_name + '_' + str(nr)
            mf = format_output(mf, lids=lids, n_pcs=n_pcs, hist=hist, ft=ft, rc=rc, rv=rv, file_name=file_name_)
            mf.loc[file_name_, 'elapsed_time(secs)'] = time.time() - start
            mf.to_csv('data/metafeatures.csv', index=True)
