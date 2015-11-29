# Code to prepare data for visual semantic embeddings
# Ramakrishna Vedantam

# *****************************************************************************
# Notes:
# *****************************************************************************
# MNLM requires:
# > Text file with a caption on each line
# > npy file with row i containing image features

# Experiment with different kinds of image and features
# Image features include:

# 1. Detection features: Only object occurrence and position
# 2. Pose + Detection features: Object pose and occurence/location
# 3. Pose + Detection + Semantic: Object expression etc.
# 4. Relative Location: Most likely a spatial pyramid of occurences?
# 5. FC7 VGG 19: FC7 features from the VGG 19 model

import json
# argument parser
import os
import sys
import argparse
# python debugger
import pdb
import numpy as np
import random
import h5py
# Python Image Library
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle


from collections import defaultdict
from nltk.tokenize import word_tokenize
from tqdm import *

import demo
# for clipart feature extraction
import abstract_features as af # TODO: change, not good for debugging

def imname(prefix, index):
    return prefix + "%012d" % (index) + '.png'

def jsonname(prefix, index):
    return prefix + "%012d" % (index) + '.json'

def compute_mean(image_folder, orig_split, indices):
    prefix = 'abstract_v002_%s2015_' % (orig_split)
    folder = os.path.join(image_folder, 'scene_img', 'img_%s2015' % (orig_split))
    mean_image = np.zeros((400,700,3), dtype=np.float32)
    # compute image mean
    for index, item in enumerate(indices):
        image_name = os.path.join(folder, imname(prefix, item))
        # remove alpha channel
        image = np.array(Image.open(image_name), dtype=np.float32)[:,:,:-1]
        mean_image += image
    mean_image /= float(len(indices))
    mean = np.mean(np.mean(mean_image, axis=0), axis=0)
    return mean

def parse_captions(json_file):
    caps = defaultdict(list)

    with open(json_file, 'r') as annfile:
        anns = json.load(annfile)['annotations']
        for item in anns:
            caps[item['image_id']].append(' '.join(word_tokenize(item['caption'])))

    return caps

def get_image_feat(feat_type, image_folder, orig_split, indices, real_split):

    feats = defaultdict(int)
    prefix = 'abstract_v002_%s2015_' % (orig_split)
    if 'fc7' in feat_type:
        # set some parameters
        folder = os.path.join(image_folder, 'scene_img', 'img_%s2015' % (orig_split)) + '/'
        print "Preparing the VGG 19 Net"
        net = demo.build_convnet()
        print "Extracting Features"
        with open('temp_{}.txt'.format(orig_split), 'w') as image_file:
            for item in tqdm(indices):
                image_file.write(imname(prefix, item) + '\n')
        image_file.close()
        feats = demo.compute_fromfile(net, 'temp_{}.txt'.format(orig_split),
                                      base_path=folder)
    elif 'hdf5' in feat_type:
        try:
            folder = os.path.join(image_folder, 'scene_img', 'img_%s2015' % (orig_split)) + '/'
            images = np.zeros((len(indices), 3, 224, 224)) # TODO: Low Priority, make general
            for index, item in tqdm(enumerate(indices)):
                images[index] = demo.load_abstract_image(folder + imname(prefix, item))
            with h5py.File('/ssd_local/rama/datasets/abstract-hdf5/{}.h5'.format(real_split), 'w') as outfile:
                outfile['images'] = images
            return True
        except:
            print "problem"
            return False
    else:
        folder = os.path.join(image_folder, 'scene_json', 'scene_%s2015_indv' % (orig_split))

        # create the abstract feature instance
        AF = pickle.load(open('extract_features/af_dump.p', 'r'))
        # TODO: Figure out a better place to initialize all this
        out_dir = '/srv/share/vqa/release_data/abstract_v002/scene_json/features_v2/'
        keep_or_remove = 'keep'
        get_names = False
        tags = feat_type
        # path to metafeature directory
        metafeat_dir = af.dir_path(os.path.join(out_dir, 'metafeatures'))

        for item in tqdm(indices):
            metafeat_fn = '{}_instances-{}.cpickle'.format(item,
                                                        AF.instance_ordering)

            cur_metafeat_fn = os.path.join(metafeat_dir,
                                        metafeat_fn)

            with open(cur_metafeat_fn, 'rb') as fp:
                cur_metafeats = pickle.load(fp)

            cur_feats, _ = AF.scene_metafeatures_to_features(cur_metafeats,
                                                            tags,
                                                            keep_or_remove,
                                                            get_names)

            feats[item] = cur_feats

    return feats


def write_vse_input(outfile, captions, feats, sp):
    """
    outfile: str (Path to directory where data is to be written, in datasets/)
    captions: dict of dict of list (captions for train, dev and test splits)
    feats: dict of dict of np.array (features for train, dev and test images)
    """
    caption_fname = open(os.path.join('/ssd_local/rama/datasets', outfile, outfile + '_' + sp + '_caps.txt'), 'w')
    im_fname = os.path.join('/ssd_local/rama/datasets', outfile, outfile + '_' + sp + '_ims.npy')

    cap_key = sp + '_captions'
    feat_key = sp + '_feats'
    # image list file
    image_features = np.float32(np.zeros([len(feats[feat_key].keys())*len(captions[cap_key].values()[0])
                                                                        , len(feats[feat_key].values()[0])]))

    fptr = 0
    for key in captions[cap_key]:
        for item in captions[cap_key][key]:
            asc_item = item.encode('ascii', 'replace')
            if asc_item != item:
                asc_item = asc_item.replace('?', ' ')
            caption_fname.write(asc_item + '\n')
            image_features[fptr, :] = np.float32(feats[feat_key][key])
            fptr += 1

    assert (fptr == image_features.shape[0])
    np.save(im_fname, image_features)
    del(image_features)
    caption_fname.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset to give as inpu\
                                     t the Visual Semantic Embedding (VSE)')

    parser.add_argument('--imfeat', default='fc7', help='What image features do \
                        you want to use?')
    parser.add_argument('--imdir',
                        default='/srv/share/vqa/release_data/abstract_v002/',
                        help='Name of the directory which contains the \
                        img_train2015 and img_val2015 folders')
    parser.add_argument('--splits', default='train-dev-test', help="What splits \
                        do you want to run feature extraction on?")
    parser.add_argument('--caption_dir', default='../datasets', help='Name of \
                        the directory which has the caption files in VQA format')
    parser.add_argument('--seed', default=123, help='Select random seed for \
                        splitting val into dev and test')

    args = parser.parse_args()

    out_dir = 'abstract-' + args.imfeat
    chk_dir= os.path.join('/ssd_local/rama/datasets', out_dir)
    # Check if output directory needs to be created
    if not os.path.exists(chk_dir):
        os.mkdir(chk_dir)

    #
    if args.imfeat != 'all':
        imfeat = set(args.imfeat.split('-'))
    else:
        imfeat = ('instance-level', 'category-general')
    # setup
    random.seed(args.seed)
    feats = {}
    captions = {}
    # parse the sentences in train and val sets of abstract scenes respectively
    train_path = os.path.join(args.caption_dir,
                              'captions_abstract_v002_train2015.json')
    val_path = os.path.join(args.caption_dir,
                            'captions_abstract_v002_val2015.json')

    captions['train_captions'] = parse_captions(train_path)
    val_captions = parse_captions(val_path)
    all_splits = {}
    # create the train, val and dev splits
    all_splits['train_split'] = set(captions['train_captions'].keys())
    val_keys = val_captions.keys()
    # shuffle entries randomly
    random.shuffle(val_keys)
    # slit half into test and half into dev
    all_splits['dev_split'] = set(val_keys[0:len(val_keys)/2])
    all_splits['test_split'] = set(val_keys[len(val_keys)/2:-1])
    assert(len(all_splits['dev_split'].intersection(
        all_splits['test_split'])) == 0)

    splits = args.splits.split('-')

    # for testing - use just one element of splits
    #all_splits['train_split'] = set([list(all_splits['train_split'])[0]])
    #all_splits['dev_split'] = set([list(all_splits['dev_split'])[0]])
    #all_splits['test_split'] = set([list(all_splits['test_split'])[0]])

    if 'hdf5' in imfeat:
        for sp in splits:
            print "Writing {} HDF5 Files".format(sp)
            if 'dev' == sp or 'test' == sp:
                orig_split = 'val'
            else:
                orig_split = 'train'
            if get_image_feat(imfeat, args.imdir, orig_split, all_splits['{}_split'.format(sp)], real_split=sp):
                print "Successfully created HDF5"
            else:
                print "Failed to create HDF5"
        sys.exit()

    captions['dev_captions'] = {k: v for k, v in val_captions.iteritems()\
                                if k in all_splits['dev_split']}
    captions['test_captions'] = {k: v for k, v in val_captions.iteritems()
                                 if k in all_splits['test_split']}
    captions['train_captions'] = {k: v for k, v in
                                  captions['train_captions'].iteritems()
                                  if k in all_splits['train_split']}

    # feats.p stores the vgg 19 features w/o mean correction
    # with open('feats.p', 'r') as readfile:
    #    feats = pickle.load(readfile)

    # decide what image features to extract
    # extract image features for dev_split

    for sp in splits:
        print "Extracting {} Set Features".format(sp)
        if 'dev' == sp or 'test' == sp:
            orig_split = 'val'
        else:
            orig_split = 'train'
        feats['{}_feats'.format(sp)] = get_image_feat(imfeat, args.imdir,
                                            orig_split,
                                            all_splits['{}_split'.format(sp)], sp)
        print "Writing the caption and image files"
        # put the captions and images in to text files and npy files respectively
        write_vse_input(out_dir, captions, feats, sp)
