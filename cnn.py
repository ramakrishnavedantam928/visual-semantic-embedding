# UTF-8
# Code to build cnn architecture of interest using lasagne

# Ramakrishna Vedantam
# Sun Nov 29 09:53:40 EST 2015

import lasagne
import cPickle as pkl
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
from lasagne.init import Glorot, Constant

def build_convnet(network='vgg19', load_param=True):
    """
    Specify the convnet architecture to use
    :params: option (str): name of the network to create
    :returns: net (dict()): lasagne network without weight initialization
    """

    print 'Building model %s ...' % (network)
    net = {}
    if network == 'vgg19':
        net['input'] = InputLayer((None, 3, 224, 224))
        net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
        net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
        net['pool1'] = PoolLayer(net['conv1_2'], 2)
        net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
        net['pool2'] = PoolLayer(net['conv2_2'], 2)
        net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
        net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
        net['pool3'] = PoolLayer(net['conv3_4'], 2)
        net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
        net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
        net['pool4'] = PoolLayer(net['conv4_4'], 2)
        net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
        net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
        net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
        net['pool5'] = PoolLayer(net['conv5_4'], 2)
        net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
        net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
        net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
        net['prob'] = NonlinearityLayer(net['fc8'], softmax)
        if load_param:
            path_to_vgg = 'models/vgg19.pkl'
            print 'Loading parameters...'
            output_layer = net['prob']
            model = pkl.load(open(path_to_vgg))
            lasagne.layers.set_all_param_values(output_layer, model['param values'])
    else:
        print "Network design not specified, specify in cnn.py"
        pass

    return net
