"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time
import argparse
import lasagne

import homogeneous_data

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model, build_sentence_encoder, build_image_encoder
from vocab import build_dictionary
from evaluation import i2t, t2i
from tools import encode_sentences, encode_images
from datasets import load_dataset

# main trainer
def trainer(data='abstract-fc7',  #f8k, f30k, coco, abstract-fc7
            margin=0.2,
            dim=1024,
            dim_image=4096,
            cnnsaveto='vse/abstract-fc7_vgg.pkl',
            dim_word=300,
            encoder='gru',  # gru OR bow
            max_epochs=15,
            dispFreq=10,
            decay_c=0.,
            grad_clip=2.,
            maxlen_w=100,
            optimizer='adam',
            batch_size = 64,
            saveto='vse/abstract-fc7.npz',
            validFreq=100,
            lrate=0.0002,
            cnn = 'vgg19',
            reload_=False):

    # Model options
    model_options = {}
    model_options['data'] = data
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_image'] = dim_image
    model_options['dim_word'] = dim_word
    model_options['encoder'] = encoder
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_
    # DONE: Added new option
    model_options['cnn'] = cnn

    print model_options

    # Load training and development sets
    print 'Loading dataset'
    # NOTE: New image features would be num_images x num_channels x h x w
    # DONE: make change in datasets / load_dataset function
    train, dev = load_dataset(data)[:2]

    # Create and save dictionary
    print 'Creating dictionary'
    worddict = build_dictionary(train[0]+dev[0])[0]
    n_words = len(worddict)
    model_options['n_words'] = n_words
    print 'Dictionary size: ' + str(n_words)
    with open('%s.dictionary.pkl'%saveto, 'wb') as f:
        pkl.dump(worddict, f)

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model'
    # NOT DOING: Make change in model / init_params L 30
    # NOT DOING: No change required in get_layer in layers.py
    # NOT DOING: Add param_init_convolution function in layers.py
    # provide an option in init params to load pretrained values
    # or use random initialization

    # all params except for CNN params initialized
    params = init_params(model_options)
    # reload parameters
    print saveto, reload_
    if reload_ and os.path.exists(saveto):
        # NOT DOING: Make changes to utils / load_params
        # NOT DOING: Combine load_params with demo / build_convnet
        # NOT DOING: Change function signature below
        params = load_params(saveto, params)
        print "Loaded VSE model parameters"

    tparams = init_tparams(params)

    # DONE: Make changes to model / build_model
    # Lines 65, 88
    # TODO: check
    trng, inps, cost, cnn_tparams, cnn_out, convnet = build_model(tparams,
                                                                  model_options)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        # cnn parameter weight decay
        for item in cnn_tparams:
            weight_decay += (item.get_value() ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Building sentence encoder'
    trng, inps_se, sentences = build_sentence_encoder(tparams, model_options)
    f_senc = theano.function(inps_se, sentences, profile=False)


    print 'Building image encoder'
    # TODO: Make changes to model / build_image_encoder
    # Line 136, L 140
    trng, inps_ie, images = build_image_encoder(tparams, model_options)
    f_ienc = theano.function(inps_ie, images, profile=False)

    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    update_cnn = lasagne.updates.adam(cost, cnn_tparams, lrate, 0.1, 0.001, 1e-8).items()
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    # not add the parameters here
    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost,
                                              update_cnn)

    print 'Optimization'

    # Each sentence in the minibatch have same length (for encoder)
    # TODO: Make changes to homogeneous_data / __init__ and homogeneous_data /
    # next
    train_iter = homogeneous_data.HomogeneousData([train[0], train[1]],
                                    batch_size=batch_size, maxlen=maxlen_w)
    dev_iter = homogeneous_data.HomogeneousData([dev[0], dev[1]],
                                    batch_size=batch_size, maxlen=maxlen_w)

    uidx = 0
    curr = 0.
    n_samples = 0

    for eidx in xrange(max_epochs):

        print 'Epoch ', eidx

        for x, im in train_iter:
            n_samples += len(x)
            uidx += 1
            # DONE: Make changes to homogeneous_data / prepare_data L 75
            x, mask, im = homogeneous_data.prepare_data(x, im, worddict, maxlen=maxlen_w, n_words=n_words)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            # Update
            ud_start = time.time()
            # prev = cnn_tparams[10].get_value()[0,0]
            # prev = tparams['ff_image_W'].get_value()[0,0]
            cost = f_grad_shared(x, mask, im)

            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            if numpy.mod(uidx, validFreq) == 0:

                print 'Computing results...'
                curr_model = {}
                curr_model['options'] = model_options
                curr_model['worddict'] = worddict
                curr_model['word_idict'] = word_idict
                curr_model['f_senc'] = f_senc
                curr_model['f_ienc'] = f_ienc

                ls = []
                lim = []
                for x_dev, im_dev in dev_iter:
                    _, _, im_dev = homogeneous_data.prepare_data(x_dev,
                            im_dev, worddict, maxlen=maxlen_w, n_words=n_words)
                    ls.extend(list(encode_sentences(curr_model, x_dev)))
                    lim.extend(list(encode_images(curr_model, im_dev)))

                ls = numpy.array(ls)
                lim = numpy.array(lim)
                (r1, r5, r10, medr) = i2t(lim, ls)
                print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
                (r1i, r5i, r10i, medri) = t2i(lim, ls)
                print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)

                currscore = r1 + r5 + r10 + r1i + r5i + r10i

                if currscore > curr:
                    curr = currscore

                    # Save model
                    print 'Saving...',
                    params = unzip(tparams)
                    # TODO: Need to add saving of VGG model / updation
                    # See if this works, and then decide what to do
                    # TODO: Might want to merge models like this, makes a lot of
                    # sense
                    numpy.savez(saveto, **params)
                    # cnn save to
                    params = lasagne.layers.get_all_param_values(convnet['fc7'])
                    pkl.dump(params, open(cnnsaveto, 'w'))
                    pkl.dump(model_options, open('%s.pkl'%saveto, 'wb'))
                    print 'Done'

        print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training code for visual semantic embedding models')
    parser.add_argument('--data', default='abstract-fc7-mean', help='Which kind of features from ./datasets do you want to use?')
    parser.add_argument('--im_dim', type=int, default=4096, help='Dimensionality of image features')
    parser.add_argument('--reload', type=bool, default=False, help='Load parameters?')
    args = parser.parse_args()

    model = os.path.join('vse-ft', args.data + '.npz')
    cnnsaveto = os.path.join('vse-ft', args.data + '_vgg' + '.pkl')
    trainer(data=args.data, dim_image=args.im_dim, saveto=model, cnnsaveto = cnnsaveto,
            reload_=args.reload, lrate=0.000005)
    print "Done"
