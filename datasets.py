"""
Dataset loading
"""
import numpy as np
import h5py

#-----------------------------------------------------------------------------#
# Specify dataset(s) location here
#-----------------------------------------------------------------------------#
path_to_data = 'datasets/'
#-----------------------------------------------------------------------------#

def load_dataset(name='abstract-fc7', load_train=True):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco, 'abstract-fc7', 'abstract-presence'

    :params: name (str): name of the dataset to use in datasets/ folder
    :params: load_train (bool): flag to indicate if training set is to be loaded

    :returns: list of tuples :
        {dataset}_caps (list of str): captions provided in the dataset
        {dataset}_ims (np.ndarray): shape (n x c x h x w): numpy array
        with preprocessed images
    """
    loc = path_to_data + name + '/'

    # Captions
    train_caps, dev_caps, test_caps = [],[],[]
    if load_train:
        with open(loc+name+'_train_caps.txt', 'rb') as f:
            for line in f:
                train_caps.append(line.strip())
    else:
        train_caps = None
    with open(loc+name+'_dev_caps.txt', 'rb') as f:
        for line in f:
            dev_caps.append(line.strip())
    with open(loc+name+'_test_caps.txt', 'rb') as f:
        for line in f:
            test_caps.append(line.strip())

    if load_train:
        # Read train images
        # dset = h5py.File('/ssd_local/rama/datasets/abstract-hdf5/{}.h5'.format('train'), 'r')['images']
        # TODO: Revert BACK!!
        dset = h5py.File('/ssd_local/rama/datasets/abstract-hdf5/{}.h5'.format('train'), 'r')['images']
        train_ims = np.zeros(dset.shape, dtype=np.float32)
        dset.read_direct(train_ims)
    else:
        train_ims = None
    # Read dev images
    # TODO: Revert BACK!!
    dset = h5py.File('/ssd_local/rama/datasets/abstract-hdf5/{}.h5'.format('dev'), 'r')['images']
    dev_ims = np.zeros(dset.shape, dtype=np.float32)
    dset.read_direct(dev_ims)
    # Read test images
    # TODO: Revert BACK!!
    dset = h5py.File('/ssd_local/rama/datasets/abstract-hdf5/{}.h5'.format('test'), 'r')['images']
    test_ims = np.zeros(dset.shape, dtype=np.float32)
    dset.read_direct(test_ims)

    return (train_caps, train_ims), (dev_caps, dev_ims), (test_caps, test_ims)

