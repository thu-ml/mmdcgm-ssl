'''
shortcuts for compsited layers
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne

from parmesan.distributions import log_stdnormal, log_normal2, log_bernoulli

import sys
sys.path.append("..")
from layers.pool_special import UnPoolLayer, UnPoolMaskLayer, MaxPoolLocationLayer, RepeatUnPoolLayer
from layers.merge import ConvConcatLayer, MLPConcatLayer

# convolutional layer
# following optional batch normalization, pooling and dropout
def convlayer(l,bn,dr,ps,n_kerns,d_kerns,nonlinearity,pad,stride,name,output_mask=False,batch_size_act=0,W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.)):
    mask = None
    l = lasagne.layers.Conv2DLayer(l, num_filters=n_kerns, filter_size=(d_kerns,d_kerns), stride=stride, pad=pad, name="Conv-"+name, W=W, b=b, nonlinearity=nonlinearity)
    if bn:
        l = lasagne.layers.batch_norm(l, name="BN-"+name)
    if ps > 1:
        if output_mask:
            mask = MaxPoolLocationLayer(l,factor=(ps,ps),batch_size=batch_size_act)
        l = lasagne.layers.MaxPool2DLayer(l, pool_size=(ps,ps), name="Pool"+name)
    if dr > 0:
        l = lasagne.layers.DropoutLayer(l, p=dr, name="Drop-"+name)
    return l, mask

# unpooling and convolutional layer
# following optional batch normalization and dropout
def unpoolconvlayer(l,bn,dr,ps,n_kerns,d_kerns,nonlinearity,pad,stride,name,type_='unpool',mask=None,W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.), noise_level=0):
    if ps > 1:
        if type_ == 'unpool':
            l = UnPoolLayer(incoming=l, factor=(ps,ps), name="UP-"+name)
        elif type_ == 'repeat':
            l = RepeatUnPoolLayer(incoming=l, factor=(ps,ps), name="UP_REP-"+name)
        elif type_ == 'unpoolmask':
            l = UnPoolMaskLayer(incoming=l, mask=mask, factor=(ps,ps), name="UP_MUSK-"+name, noise_level=noise_level)
    l = lasagne.layers.Conv2DLayer(l, num_filters=n_kerns, filter_size=(d_kerns,d_kerns), stride=stride, pad=pad, name="Conv-"+name, W=W, b=b, nonlinearity=nonlinearity)
    if bn:
        l = lasagne.layers.batch_norm(l, name="BN-"+name)
    if dr > 0:
        l = lasagne.layers.DropoutLayer(l, p=dr, name="Drop-"+name)
    return l

# fractional strided convolutional layer
# following optional batch normalization and dropout
def fractionalstridedlayer(l,bn,dr,n_kerns,d_kerns,nonlinearity,pad,stride,name,W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.)):
    # print bn,dr,n_kerns,d_kerns,nonlinearity,pad,stride,name
    l = lasagne.layers.TransposedConv2DLayer(l, num_filters=n_kerns, filter_size=(d_kerns,d_kerns), stride=stride, crop=pad, name="FS_Conv-"+name, W=W, b=b, nonlinearity=nonlinearity)
    if bn:
        l = lasagne.layers.batch_norm(l, name="BN-"+name)
    if dr > 0:
        l = lasagne.layers.DropoutLayer(l, p=dr, name="Drop-"+name)
    return l

# mlp layer
# following optional batch normalization and dropout
def mlplayer(l,bn,dr,num_units,nonlinearity,name):
    l = lasagne.layers.DenseLayer(l,num_units=num_units,nonlinearity=nonlinearity,name="MLP-"+name)
    if bn:
        l = lasagne.layers.batch_norm(l, name="BN-"+name)
    if dr > 0:
        l = lasagne.layers.DropoutLayer(l, p=dr, name="Drop-"+name)
    return l
