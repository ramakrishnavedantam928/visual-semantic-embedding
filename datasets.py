"""
Dataset loading
"""
import utils
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
    if 'mini' in name.split('-'):
        hdf5_loc = '/ssd_local/rama/datasets/abstract-mini-hdf5/{}.h5'
    else:
        hdf5_loc = '/ssd_local/rama/datasets/abstract-hdf5/{}.h5'


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
        dset = h5py.File(hdf5_loc.format('train'), 'r')['images']
        print "Loaded Train"
        train_ims = utils.repeat_list(dset, 5)
        print "Repeating Train"
    else:
        train_ims = None


    # Read dev images
    dset = h5py.File(hdf5_loc.format('dev'), 'r')['images']
    print "Loaded Dev"
    dev_ims = utils.repeat_list(dset, 5)
    print "Repeating Dev"


    # Read test images
    dset = h5py.File(hdf5_loc.format('test'), 'r')['images']
    print "Loaded Test"
    test_ims = utils.repeat_list(dset, 5)
    print "Repeating Test"

    return (train_caps, train_ims), (dev_caps, dev_ims), (test_caps, test_ims)
