import time
import h5py
import os
import numpy as np
from progressbar import ProgressBar
from . import config

start = time.time()

direc = config.config.get('gallearn_paths', 'firebox_data_dir')
firebox_snap = config.config.get('gallearn_paths', 'firebox_snap')

def get_N_structures(obj_num):
    obj_num = str(obj_num)
    fname = direc + 'ahf_objects_1200/ahf_object_' + obj_num + '.hdf5'
    with h5py.File(fname, 'r') as f:
        stars_in_gal = f['particleIDs'][:]

    fname = os.path.join(direc, firebox_snap, 'object_' + obj_num + '.hdf5')
    with h5py.File(fname, 'r') as f:
        all_stars = f['stars_id'][:]

    in_gal = np.isin(all_stars, stars_in_gal)
    N_bound = in_gal.sum()
    N_stars = all_stars.shape[0]
    frac = N_bound / N_stars
    
    return frac, N_stars, N_bound

obj_nums = []
N_starss = []
N_bounds = []
fracs = []

files = os.listdir(os.path.join(direc, firebox_snap))
pbar = ProgressBar()
for ahf_fname in pbar(files):
    ibeg = ahf_fname.rindex('_') + 1
    iend = ahf_fname.rindex('.hdf5')
    obj_num = ahf_fname[ibeg : iend]
    result = get_N_structures(obj_num)
    obj_nums += [obj_num]
    fracs += [result[0]]
    N_starss += [result[1]]
    N_bounds += [result[2]]


end = time.time()
print(str(end - start) + 'seconds')
