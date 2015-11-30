import numpy
import copy
import sys

class HomogeneousData():

    def __init__(self, data, batch_size=128, maxlen=None):
        self.batch_size = 128
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen

        self.prepare()
        self.reset()

    def prepare(self):
        self.caps = self.data[0]
        # DONE: Name this as Images more appropriate
        self.images = self.data[1]

        # find the unique lengths
        self.lengths = [len(cc.split()) for cc in self.caps]
        self.len_unique = numpy.unique(self.lengths)
        # remove any overly long sentences
        if self.maxlen:
            self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]

        # indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

        # current counter
        self.len_curr_counts = copy.copy(self.len_counts)

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size

        caps = [self.caps[ii] for ii in curr_indices]
        # DONE: Name feats as Images
        images = [self.images[ii] for ii in curr_indices]

        # DONE: Name feats as Images
        return caps, images

    def __iter__(self):
        return self

# TODO: Change features to images below
def prepare_data(caps, images, worddict, maxlen=None, n_words=10000):
    """
    Put data into format useable by the model
    """
    seqs = []
    # TODO: change feat_list to image_list
    image_list = []
    for i, cc in enumerate(caps):
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])
        # TODO:  Change features to images below
        image_list.append(images[i])

    lengths = [len(s) for s in seqs]

    if maxlen != None and numpy.max(lengths) >= maxlen:
        new_seqs = []
        # TODO: Rename to new_image_list
        new_ew_image_list = []
        new_lengths = []
        # TODO: Rename to image_list below
        for l, s, y in zip(lengths, seqs, image_list):
            if l < maxlen:
                new_seqs.append(s)
                # TODO: Rename to new_image_list
                new_image_list.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        # TODO: Rename to new_image_list and image_list
        image_list = new_image_list
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    # TODO: Rename to image_list, and change dimensions of y to n x c x h x w
    y = numpy.zeros((len(image_list), image_list[0].shape[0],
                     image_list[0].shape[1],
                     image_list[0].shape[2])).astype('float32')
    for idx, ff in enumerate(image_list):
        y[idx,:] = ff

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask, y

