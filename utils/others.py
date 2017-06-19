import shutil, gzip, os, cPickle, time, math, operator, argparse

import numpy as np
import theano.tensor as T
import theano, lasagne


def get_pad(pad):
    if pad not in ['same', 'valid', 'full']:
        pad = tuple(map(int, pad.split('-')))
    return pad

def get_pad_list(pad_list):
    re_list = []
    for p in pad_list:
        re_list.append(get_pad(p))
    return re_list

# nonlinearities
def get_nonlin(nonlin):
    if nonlin == 'rectify':
        return lasagne.nonlinearities.rectify
    elif nonlin == 'leaky_rectify':
        return lasagne.nonlinearities.LeakyRectify(0.1)
    elif nonlin == 'tanh':
        return lasagne.nonlinearities.tanh
    elif nonlin == 'sigmoid':
        return lasagne.nonlinearities.sigmoid
    elif nonlin == 'maxout':
        return 'maxout'
    elif nonlin == 'none':
        return lasagne.nonlinearities.identity
    else:
        raise ValueError('invalid non-linearity \'' + nonlin + '\'')
def get_nonlin_list(nonlin_list):
    re_list = []
    for n in nonlin_list:
        re_list.append(get_nonlin(n))
    return re_list

def bernoullisample(x):
    return np.random.binomial(1,x,size=x.shape).astype(theano.config.floatX)

def build_log_file(args, filename_script, extra=None):
    res_out = args.outfolder
    res_out += '_'
    res_out += args.name
    res_out += '_'
    if extra is not None:
        res_out += extra
        res_out += '_' 
    res_out += str(int(time.time()))
    if not os.path.exists(res_out):
        os.makedirs(res_out)

    # write commandline parameters to header of logfile
    args_dict = vars(args)
    sorted_args = sorted(args_dict.items(), key=operator.itemgetter(0))
    description = []
    description.append('######################################################')
    description.append('# --Commandline Params--')
    for name, val in sorted_args:
        description.append("# " + name + ":\t" + str(val))
    description.append('######################################################')
    
    logfile = os.path.join(res_out, 'logfile.log')
    model_out = os.path.join(res_out, 'model')
    with open(logfile,'w') as f:
        for l in description:
            f.write(l + '\n')
    return logfile, res_out

def array2file_2D(array,logfile):
    assert len(array.shape) == 2, array.shape
    with open(logfile,'a') as f:
       for i in xrange(array.shape[0]):
        for j in xrange(array.shape[1]):
            f.write(str(array[i][j])+' ')
        f.write('\n')

def printarray_2D(array, precise=2):
    assert len(array.shape) == 2, array.shape
    format = '%.'+str(precise)+'f'
    for i in xrange(array.shape[0]):
        for j in xrange(array.shape[1]):
            print format %array[i][j],
        print