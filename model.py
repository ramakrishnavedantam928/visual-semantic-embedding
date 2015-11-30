"""
Model specification
"""
import theano
import theano.tensor as tensor
from theano.tensor.extra_ops import fill_diagonal
import numpy

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import _p, ortho_weight, norm_weight, xavier_weight, tanh, l2norm
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer

def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()

    # Word embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # Sentence encoder
    if options['encoder'] != 'bow':
        params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                                  nin=options['dim_word'], dim=options['dim'])

    # Change this call for image fintetuning
    params = get_layer('ff')[0](options, params, prefix='ff_image', nin=options['dim_image'], nout=options['dim'])
    # Parameters for the CNN are initialized automatically by lasagne
    # So just get the trainable parameters here
    # Cant get those without network definition, so skip cnn parameters here.

    return params

def contrastive_loss(margin, im, s):
    """
    Compute contrastive loss
    """
    # compute image-sentence score matrix
    scores = tensor.dot(im, s.T)
    diagonal = scores.diagonal()

    # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
    cost_s = tensor.maximum(0, margin - diagonal + scores)
    # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
    cost_im = tensor.maximum(0, margin - diagonal.reshape((-1, 1)) + scores)

    # clear diagonals
    cost_s = fill_diagonal(cost_s, 0)
    cost_im = fill_diagonal(cost_im, 0)

    return cost_s.sum() + cost_im.sum()

def build_model(tparams, options):
    """
    Computation graph for the model
    """
    opt_ret = dict()
    trng = RandomStreams(1234)

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype='float32')

    # make this a 4D matrix since images will be input
    im = tensor.tensor4('im', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples,
                                                options['dim_word']])

    # Encode sentences (source)
    if options['encoder'] == 'bow':
        sents = (emb * mask[:,:,None]).sum(0)
    else:
        proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                                prefix='encoder',
                                                mask=mask)
        sents = proj[0][-1]
    sents = l2norm(sents)

    # TODO: Encode images (source)
    # Update the expression below based on implementation
    # in layers / convff
    # pass the image through the CNN
    im_feat, cnn_tparams, cnn_rparams = get_layer('cnn')[0](options, im,
                                                            test=False)
    # pass features through image embedding
    images = get_layer('ff')[1](tparams, im_feat, options, prefix='ff_image',
                                activ='linear')

    # Compute loss
    cost = contrastive_loss(options['margin'], images, sents)

    return trng, [x, mask, im], cost, cnn_tparams

def build_sentence_encoder(tparams, options):
    """
    Encoder only, for sentences
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # Word embedding
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encode sentences
    if options['encoder'] == 'bow':
        sents = (emb * mask[:,:,None]).sum(0)
    else:
        proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                                prefix='encoder',
                                                mask=mask)
        sents = proj[0][-1]
    sents = l2norm(sents)

    return trng, [x, mask], sents

def build_image_encoder(tparams, options):
    """
    Encoder only, for images
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # image features
    # TODO: make the same change you made above in build model
    im = tensor.tensor4('im', dtype='float32')

    # Encode images
    im_feat, cnn_tparams, cnn_rparams = get_layer('cnn')[0](options, im,
                                                            test=True)
    # pass features through image embedding
    images = get_layer('ff')[1](tparams, im_feat, options, prefix='ff_image',
                                activ='linear')

   # TODO: Make the same change you made above in build model
    images = l2norm(images)

    return trng, [im], images


