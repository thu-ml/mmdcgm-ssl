'''
This code implements max-margin deep conditional generative model which incorporates the side information in generative modelling and uses a semi-supervised classifier to infer the latent labels
'''

import gzip, os, cPickle, time, math, argparse, shutil, sys

import numpy as np
import theano.tensor as T
import theano
import lasagne
from parmesan.datasets import load_mnist_realval, load_mnist_binarized, load_frey_faces, load_norb_small
from datasets import load_cifar10, load_svhn
from datasets_norb import load_numpy_subclasses
from parmesan.layers import SampleLayer

from layers.merge import ConvConcatLayer, MLPConcatLayer
from utils.others import get_nonlin_list, get_pad_list, bernoullisample, build_log_file, printarray_2D, array2file_2D
from components.shortcuts import convlayer, fractionalstridedlayer, unpoolconvlayer, mlplayer
from components.objectives import latent_gaussian_x_gaussian, latent_gaussian_x_bernoulli
from components.objectives import multiclass_s3vm_loss, multiclass_hinge_loss
from utils.create_ssl_data import create_ssl_data, create_ssl_data_subset
import  utils.paramgraphics as paramgraphics

'''
parameters
'''
# global
theano.config.floatX = 'float32'
filename_script = os.path.basename(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, default="mnist_real")
parser.add_argument("-outfolder", type=str, default=os.path.join("results-ssl", os.path.splitext(filename_script)[0]))
parser.add_argument("-preprocess", type=str, default="none")
parser.add_argument("-subset_flag", type=str, default ='false')
# architecture
parser.add_argument("-nz", type=int, default=100)
parser.add_argument("-batch_norm_dgm", type=str, default='false')
parser.add_argument("-top_mlp", type=str, default='false')
parser.add_argument("-mlp_size", type=int, default=256)
parser.add_argument("-batch_norm_classifier", type=str, default='false')
# classifier
parser.add_argument("-num_labelled", type=int, default=100)
parser.add_argument("-num_labelled_per_batch", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=200)
parser.add_argument("-delta", type=float, default=1.)
parser.add_argument("-alpha_decay", type=float, default=1e-4)
parser.add_argument("-alpha_hinge", type=float, default=1.)
parser.add_argument("-alpha_hat", type=float, default=.3)
parser.add_argument("-alpha_reg", type=float, default=0)
parser.add_argument("-alpha", type=float, default=.1)
parser.add_argument("-norm_type", type=int, default=2)
parser.add_argument("-form", type=str, default='mean_class')
# feature extractor
parser.add_argument("-nlayers_cla", type=int, default=3)
parser.add_argument("-nk_cla", type=str, default='32,64,128')
parser.add_argument("-dk_cla", type=str, default='4,5,3')
parser.add_argument("-pad_cla", type=str, default='valid,valid,valid')
parser.add_argument("-str_cla", type=str, default='2,2,2')
parser.add_argument("-ps_cla", type=str, default='1,1,1')
parser.add_argument("-nonlin_cla", type=str, default='rectify,rectify,rectify')
parser.add_argument("-dr_cla", type=str, default='0,0,0')
# encoder
parser.add_argument("-nlayers_enc", type=int, default=3)
parser.add_argument("-nk_enc", type=str, default='32,64,128')
parser.add_argument("-dk_enc", type=str, default='4,5,3')
parser.add_argument("-pad_enc", type=str, default='valid,valid,valid')
parser.add_argument("-str_enc", type=str, default='2,2,2')
parser.add_argument("-ps_enc", type=str, default='1,1,1')
parser.add_argument("-nonlin_enc", type=str, default='rectify,rectify,rectify')
parser.add_argument("-dr_enc", type=str, default='0,0,0')
# decoder
parser.add_argument("-nlayers_dec", type=int, default=4)
parser.add_argument("-nk_dec", type=str, default='128,64,32,1')
parser.add_argument("-dk_dec", type=str, default='3,5,4,5')
parser.add_argument("-pad_dec", type=str, default='valid,valid,valid,same')
parser.add_argument("-str_dec", type=str, default='2,2,2,1')
parser.add_argument("-up_method", type=str, default='frac_strided,frac_strided,frac_strided,none')
parser.add_argument("-ps_dec", type=str, default='1,1,1,1')
parser.add_argument("-nonlin_dec", type=str, default='rectify,rectify,rectify,sigmoid')
parser.add_argument("-dr_dec", type=str, default='0,0,0,0')
# optimization
parser.add_argument("-flag", type=str, default='validation') # validation for anneal learning rate
parser.add_argument("-ssl_data_seed", type=int, default=0) # random seed for ssl data generation
parser.add_argument("-lr", type=float, default=0.0003)
parser.add_argument("-nepochs", type=int, default=200)
parser.add_argument("-anneal_lr_epoch", type=int, default=100)
parser.add_argument("-anneal_lr_factor", type=float, default=.99)
parser.add_argument("-every_anneal", type=int, default=1)
clip_grad = 1
max_norm = 5
# name
parser.add_argument("-name", type=str, default='')
# inference
parser.add_argument("-eq_samples", type=int,
        help="number of samples for the expectation over q(z|x)", default=1)
parser.add_argument("-iw_samples", type=int,
        help="number of importance weighted samples", default=1)

# random seeds for reproducibility
np.random.seed(1234)
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=1234)

# get parameters
# global
args = parser.parse_args()
dataset = args.dataset
subset_flag = args.subset_flag == 'true' or args.subset_flag == 'True'
eval_epoch = 1
# architecture
nz = args.nz
bn_dgm = args.batch_norm_dgm == 'true' or args.batch_norm_dgm == 'True'
top_mlp = args.top_mlp == 'true' or args.top_mlp == 'True'
mlp_size = args.mlp_size
bn_cla = args.batch_norm_classifier == 'true' or args.batch_norm_classifier == 'True'
# classifier
num_labelled = args.num_labelled
batch_size = args.batch_size
num_labelled_per_batch = args.num_labelled_per_batch
assert num_labelled % num_labelled_per_batch == 0
delta = args.delta
alpha_decay = args.alpha_decay
alpha_hinge = args.alpha_hinge
alpha_reg = args.alpha_reg
alpha_hat = args.alpha_hat
alpha = args.alpha
norm_type = args.norm_type
form = args.form
# feature extractor
nlayers_cla = args.nlayers_cla
nk_cla = map(int, args.nk_cla.split(','))
dk_cla = map(int, args.dk_cla.split(','))
pad_cla = map(str, args.pad_cla.split(','))
str_cla = map(int, args.str_cla.split(','))
ps_cla = map(int, args.ps_cla.split(','))
dr_cla = map(float, args.dr_cla.split(','))
nonlin_cla = get_nonlin_list(map(str, args.nonlin_cla.split(',')))
# encoder
nlayers_enc = args.nlayers_enc
nk_enc = map(int, args.nk_enc.split(','))
dk_enc = map(int, args.dk_enc.split(','))
pad_enc = get_pad_list(map(str, args.pad_enc.split(',')))
str_enc = map(int, args.str_enc.split(','))
ps_enc = map(int, args.ps_enc.split(','))
dr_enc = map(float, args.dr_enc.split(','))
nonlin_enc = get_nonlin_list(map(str, args.nonlin_enc.split(',')))
# decoder
nlayers_dec = args.nlayers_dec
nk_dec = map(int, args.nk_dec.split(','))
dk_dec = map(int, args.dk_dec.split(','))
pad_dec = get_pad_list(map(str, args.pad_dec.split(',')))
str_dec = map(int, args.str_dec.split(','))
ps_dec = map(int, args.ps_dec.split(','))
dr_dec = map(float, args.dr_dec.split(','))
nonlin_dec = get_nonlin_list(map(str, args.nonlin_dec.split(',')))
up_method = map(str, args.up_method.split(','))
# optimization
flag = args.flag
ssl_data_seed = args.ssl_data_seed
if ssl_data_seed == -1:
    ssl_data_seed = int(time.time())
lr = args.lr
num_epochs = args.nepochs
anneal_lr_epoch = args.anneal_lr_epoch
anneal_lr_factor = args.anneal_lr_factor
every_anneal = args.every_anneal
# inference
iw_samples = args.iw_samples
eq_samples = args.eq_samples
# log file
logfile, res_out = build_log_file(args, filename_script)
shutil.copy(os.path.realpath(__file__), os.path.join(res_out, filename_script))

'''
datasets
'''
if dataset == 'mnist_real':
    colorImg = False
    dim_input = (28,28)
    in_channels = 1
    num_classes = 10
    generation_scale = False
    num_generation = num_classes*num_classes
    vis_epoch = 100
    distribution = 'bernoulli'
    num_features = in_channels*dim_input[0]*dim_input[1]
    print "Using real-valued mnist dataset"
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_mnist_realval()
    if flag == 'validation':
        test_x = valid_x
        test_t = valid_t
    else:
        train_x = np.concatenate([train_x,valid_x])
        train_t = np.hstack((train_t, valid_t))
    train_x_size = train_t.shape[0]
    train_t = np.int32(train_t)
    test_t = np.int32(test_t)
    train_x = train_x.astype(theano.config.floatX)
    test_x = test_x.astype(theano.config.floatX)
    train_x = train_x.reshape((-1, in_channels)+dim_input)
    test_x = test_x.reshape((-1, in_channels)+dim_input)
    # prepare data for semi-supervised learning
    if subset_flag:
        # instead of sampling from 60000 data, sample 100 data for 10 times to make sure that the labelled data with smaller size is a subset of that with larger size. 
        x_labelled, y_labelled, x_unlabelled, _ = create_ssl_data_subset(train_x, train_t, num_classes, num_labelled, 100, ssl_data_seed)
    else:
        x_labelled, y_labelled, x_unlabelled, _ = create_ssl_data(train_x, train_t, num_classes, num_labelled, ssl_data_seed)
    y_labelled = np.int32(y_labelled)   
elif dataset == 'cifar10':
    colorImg = True
    dim_input = (32,32)
    in_channels = 3
    num_classes = 10
    generation_scale = False
    num_generation = num_classes*num_classes
    vis_epoch = 100
    distribution = 'bernoulli'
    num_features = in_channels*dim_input[0]*dim_input[1]
    print "Using cifar10 dataset"
    train_x, train_t, valid_x, valid_t, test_x, test_t = load_cifar10(num_val=5000, normalized=True, centered=True)
    if flag == 'validation':
        test_x = valid_x
        test_t = valid_t
    else:
        train_x = np.concatenate([train_x,valid_x])
        train_t = np.hstack((train_t, valid_t))
    train_x_size = train_t.shape[0]
    train_t = np.int32(train_t)
    test_t = np.int32(test_t)
    train_x = train_x.astype(theano.config.floatX)
    test_x = test_x.astype(theano.config.floatX)
    train_x = train_x.reshape((-1, in_channels)+dim_input)
    test_x = test_x.reshape((-1, in_channels)+dim_input)
    # prepare data for semi-supervised learning
    x_labelled, y_labelled, x_unlabelled, _ = create_ssl_data(train_x, train_t, num_classes, num_labelled, ssl_data_seed)
    y_labelled = np.int32(y_labelled)
elif dataset == 'svhn':
    colorImg = True
    dim_input = (32,32)
    in_channels = 3
    num_classes = 10
    generation_scale = False
    num_generation = num_classes*num_classes
    vis_epoch = 10
    distribution = 'bernoulli'
    num_features = in_channels*dim_input[0]*dim_input[1]
    print "Using svhn dataset"
    train_x, train_t, valid_x, valid_t, test_x, test_t, avg = load_svhn(normalized=True, centered=False)
    if flag == 'validation':
        test_x = valid_x
        test_t = valid_t
    else:
        train_x = np.concatenate([train_x,valid_x])
        train_t = np.hstack((train_t, valid_t))
    train_x_size = train_t.shape[0]
    train_t = np.int32(train_t)
    test_t = np.int32(test_t)
    train_x = train_x.astype(theano.config.floatX)
    test_x = test_x.astype(theano.config.floatX)
    train_x = train_x.reshape((-1, in_channels)+dim_input)
    test_x = test_x.reshape((-1, in_channels)+dim_input)
    # prepare data for semi-supervised learning
    x_labelled, y_labelled, x_unlabelled, _ = create_ssl_data(train_x, train_t, num_classes, num_labelled, ssl_data_seed)
    y_labelled = np.int32(y_labelled)
elif dataset == 'norb':
    colorImg = False
    dim_input = (32,32)
    in_channels = 1
    num_classes = 5
    generation_scale = False
    num_generation = num_classes*num_classes
    vis_epoch = 100
    distribution = 'bernoulli'
    num_features = in_channels*dim_input[0]*dim_input[1]
    print "Using small norb dataset"
    x, t = load_numpy_subclasses(size=dim_input[0], normalize=True, centered=False)
    x = np.transpose(x)
    t = t.flatten()
    train_x = x[:24300]
    test_x = x[24300*2:24300*3]
    train_t = t[:24300]
    test_t = t[24300*2:24300*3]
    if flag == 'validation':
        test_x = train_x[:1000]
        test_t = train_t[:1000]
        train_x = train_x[1000:]
        train_t = train_t[1000:]
    train_x_size = train_t.shape[0]
    train_t = np.int32(train_t)
    test_t = np.int32(test_t)
    train_x = train_x.astype(theano.config.floatX)
    test_x = test_x.astype(theano.config.floatX)
    train_x = train_x.reshape((-1, in_channels)+dim_input)
    test_x = test_x.reshape((-1, in_channels)+dim_input)
    # prepare data for semi-supervised learning
    x_labelled, y_labelled, x_unlabelled, _ = create_ssl_data(train_x, train_t, num_classes, num_labelled, ssl_data_seed)
    y_labelled = np.int32(y_labelled)

# preprocess
if args.preprocess == 'none':
    preprocesses_dataset = None
elif args.preprocess == 'bernoullisample':
    preprocesses_dataset = bernoullisample
elif args.preprocess == 'dequantify':
    pass

# shared variables for semi-supervised learning
sh_x_train_labelled = theano.shared(x_labelled, borrow=True)
sh_x_train_unlabelled = theano.shared(x_unlabelled, borrow=True)
sh_t_train_labelled = theano.shared(y_labelled, borrow=True)
sh_x_test = theano.shared(test_x, borrow=True)
sh_t_test = theano.shared(test_t, borrow=True)
if preprocesses_dataset is not None:
    sh_x_train_labelled_preprocessed = theano.shared(preprocesses_dataset(x_labelled), borrow=True)
    sh_x_train_unlabelled_preprocessed = theano.shared(preprocesses_dataset(x_unlabelled), borrow=True)
    sh_x_test_preprocessed = theano.shared(preprocesses_dataset(test_x), borrow=True)

# visualize labeled data
if True:
    print 'size of training data ', x_labelled.shape, y_labelled.shape, x_unlabelled.shape
    _x_mean = x_labelled.reshape((num_labelled,-1))
    _x_mean = _x_mean[:num_generation]
    y_order = np.argsort(y_labelled[:num_generation])
    _x_mean = _x_mean[y_order]
    image = paramgraphics.mat_to_img(_x_mean.T, dim_input, colorImg=colorImg, scale=generation_scale, 
        save_path=os.path.join(res_out, 'labeled_data'+str(ssl_data_seed)+'.png'))

'''
building block
'''
# shortcuts
encodelayer = convlayer

# decoder layer
def decodelayer(l,up_method,bn,dr,ps,n_kerns,d_kerns,nonlinearity,pad,stride,name):
    # upsampling
    if up_method == 'unpool':
        h_g = unpoolconvlayer(l,bn,dr,ps,n_kerns,d_kerns,nonlinearity,pad,stride,name,'unpool',None)
    elif up_method == 'frac_strided':
        h_g = fractionalstridedlayer(l,bn,dr,n_kerns,d_kerns,nonlinearity,pad,stride,name)
    elif up_method == 'none':
        h_g, _ = convlayer(l,bn,dr,ps,n_kerns,d_kerns,nonlinearity,pad,stride,name)
    else:
        raise Exception('Unknown upsampling method')
    return h_g


'''
model
'''
# symbolic variables
sym_iw_samples = T.iscalar('iw_samples')
sym_eq_samples = T.iscalar('eq_samples')
sym_lr = T.scalar('lr')
sym_x = T.tensor4('x')
sym_x_cla = T.tensor4('x_cla')
sym_y = T.ivector('y')
sym_index = T.iscalar('index')
sym_batch_size = T.iscalar('batch_size')
batch_slice = slice(sym_index * sym_batch_size, (sym_index + 1) * sym_batch_size)
sym_index_l = T.iscalar('index_l')
sym_index_u = T.iscalar('index_u')
sym_batch_size_l = T.iscalar('batch_size_l')
sym_batch_size_u = T.iscalar('batch_size_u')
batch_slice_l = slice(sym_index_l * sym_batch_size_l, (sym_index_l + 1) * sym_batch_size_l)
batch_slice_u = slice(sym_index_u * sym_batch_size_u, (sym_index_u + 1) * sym_batch_size_u)

# x2y
l_in_x_cla = lasagne.layers.InputLayer((None, in_channels)+dim_input)
l_cla = [l_in_x_cla,]
print lasagne.layers.get_output_shape(l_cla[-1])
# conv layers
for i in xrange(nlayers_cla):
    l, _= convlayer(l_cla[-1],bn_cla,dr_cla[i],ps_cla[i],nk_cla[i],dk_cla[i],nonlin_cla[i],pad_cla[i],str_cla[i],'CLA-'+str(i+1))
    l_cla.append(l)
    print lasagne.layers.get_output_shape(l_cla[-1])

# feature and classifier
if top_mlp:
    l_cla.append(lasagne.layers.FlattenLayer(l_cla[-1]))
    feature = mlplayer(l_cla[-1],bn_cla,0.5,mlp_size,lasagne.nonlinearities.rectify,name='MLP-CLA')
else:
    feature = lasagne.layers.GlobalPoolLayer(l_cla[-1])
classifier = lasagne.layers.DenseLayer(feature, num_units=num_classes, nonlinearity=lasagne.nonlinearities.identity, W=lasagne.init.Normal(1e-2, 0), name="classifier")

# encoder xy2z
l_in_x = lasagne.layers.InputLayer((None, in_channels)+dim_input)
l_in_y = lasagne.layers.InputLayer((None,))
l_enc = [l_in_x,]
for i in xrange(nlayers_enc):
    l_enc.append(ConvConcatLayer([l_enc[-1], l_in_y], num_classes))
    l, _ = encodelayer(l_enc[-1],bn_dgm,dr_enc[i],ps_enc[i],nk_enc[i],dk_enc[i],nonlin_enc[i],pad_enc[i],str_enc[i],'ENC-'+str(i+1),False,0)
    l_enc.append(l)
    print lasagne.layers.get_output_shape(l_enc[-1])

# reshape
after_conv_shape = lasagne.layers.get_output_shape(l_enc[-1])
after_conv_size = int(np.prod(after_conv_shape[1:]))
l_enc.append(lasagne.layers.FlattenLayer(l_enc[-1]))
print lasagne.layers.get_output_shape(l_enc[-1])

# compute parameters and sample z
l_mu = mlplayer(l_enc[-1],False,0,nz,lasagne.nonlinearities.identity,'ENC-MU')
l_log_var = mlplayer(l_enc[-1],False,0,nz,lasagne.nonlinearities.identity,'ENC-LOG_VAR')
l_z = SampleLayer(mean=l_mu,log_var=l_log_var,eq_samples=sym_eq_samples,iw_samples=sym_iw_samples)

# decoder zy2x
l_dec = [l_z,]
print lasagne.layers.get_output_shape(l_dec[-1])

# reshape
l_dec.append(mlplayer(l_dec[-1],bn_dgm,0,after_conv_size,lasagne.nonlinearities.rectify, 'DEC_l_Z'))
print lasagne.layers.get_output_shape(l_dec[-1])
l_dec.append(lasagne.layers.ReshapeLayer(l_dec[-1], shape=(-1,)+after_conv_shape[1:]))
print lasagne.layers.get_output_shape(l_dec[-1])
for i in (xrange(nlayers_dec-1)):
    l_dec.append(ConvConcatLayer([l_dec[-1], l_in_y], num_classes))
    l = decodelayer(l_dec[-1],up_method[i],bn_dgm,dr_dec[i],ps_dec[i],nk_dec[i],dk_dec[i],nonlin_dec[i],pad_dec[i],str_dec[i],'DEC-'+str(i+1))
    l_dec.append(l)
    print lasagne.layers.get_output_shape(l_dec[-1])

# mu and var
if distribution == 'gaussian':
    l_dec_x_mu = decodelayer(l_dec[-1],up_method[-1],bn_dgm,dr_dec[-1],ps_dec[-1],nk_dec[-1],dk_dec[-1],lasagne.nonlinearities.sigmoid,pad_dec[-1],str_dec[-1],'DEC-MU')
    l_dec_x_log_var = decodelayer(l_dec[-1],up_method[-1],bn_dgm,dr_dec[-1],ps_dec[-1],nk_dec[-1],dk_dec[-1],lasagne.nonlinearities.identity,pad_dec[-1],str_dec[-1],'DEC-LOG_VAR')
elif distribution == 'bernoulli':
    l_dec_x_mu = decodelayer(l_dec[-1],up_method[-1],bn_dgm,dr_dec[-1],ps_dec[-1],nk_dec[-1],dk_dec[-1],lasagne.nonlinearities.sigmoid,pad_dec[-1],str_dec[-1],'DEC-MU')
print lasagne.layers.get_output_shape(l_dec_x_mu)

# predictions and accuracies 
predictions_train = lasagne.layers.get_output(classifier, sym_x_cla, deterministic=False)
predictions_eval = lasagne.layers.get_output(classifier, sym_x_cla, deterministic=True)
accurracy_train_labeled = lasagne.objectives.categorical_accuracy(predictions_train[:sym_batch_size_l], sym_y)
accurracy_eval = lasagne.objectives.categorical_accuracy(predictions_eval, sym_y)

# weight decays
weight_decay_classifier = lasagne.regularization.regularize_layer_params_weighted({classifier:1}, lasagne.regularization.l2)


'''
learning
'''
# discriminative objective
classifier_cost_train = multiclass_s3vm_loss(predictions=predictions_train, targets=sym_y, weight_decay=weight_decay_classifier, norm_type=norm_type, form=form, num_labelled=sym_batch_size_l, alpha_decay=alpha_decay, alpha_reg=alpha_reg, alpha_hat=alpha_hat, alpha_hinge=alpha_hinge, delta=delta)
classifier_cost_eval = multiclass_hinge_loss(predictions=predictions_eval, targets=sym_y, weight_decay=weight_decay_classifier, alpha_decay=alpha_decay) # no hat loss for testing

cost_cla = classifier_cost_train

# generative objective
predictions_train_hard = predictions_train.argmax(axis=1)
predictions_eval_hard = predictions_eval.argmax(axis=1)

if distribution == 'bernoulli':
    z_train, z_mu_train, z_log_var_train, x_mu_train = lasagne.layers.get_output([l_z, l_mu, l_log_var, l_dec_x_mu], {l_in_x:sym_x, l_in_y:T.concatenate([sym_y,predictions_train_hard[sym_batch_size_l:]], axis=0)}, deterministic=False)
    z_eval, z_mu_eval, z_log_var_eval, x_mu_eval = lasagne.layers.get_output([l_z, l_mu, l_log_var, l_dec_x_mu], {l_in_x:sym_x, l_in_y:predictions_eval_hard}, deterministic=True)

    # lower bounds
    LL_train, log_qz_given_xy_train, log_pz_train, log_px_given_zy_train = latent_gaussian_x_bernoulli(z_train, z_mu_train, z_log_var_train, x_mu_train, sym_x, latent_size=nz, num_features=num_features, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)
    LL_eval, log_qz_given_xy_eval, log_pz_eval, log_px_given_zy_eval = latent_gaussian_x_bernoulli(z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, sym_x, latent_size=nz, num_features=num_features, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

elif distribution == 'gaussian':
    z_train, z_mu_train, z_log_var_train, x_mu_train, x_log_var_train = lasagne.layers.get_output([l_z, l_mu, l_log_var, l_dec_x_mu, l_dec_x_log_var], {l_in_x:sym_x, l_in_y:T.concatenate([sym_y,predictions_train_hard[sym_batch_size_l:]], axis=0)}, deterministic=False)
    z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, x_log_var_eval = lasagne.layers.get_output([l_z, l_mu, l_log_var, l_dec_x_mu, l_dec_x_log_var], {l_in_x:sym_x,l_in_y:predictions_eval_hard}, deterministic=True)

    LL_train, log_qz_given_xy_train, log_pz_train, log_px_given_zy_train = latent_gaussian_x_gaussian(z_train, z_mu_train, z_log_var_train, x_mu_train, x_log_var_train, sym_x, latent_size=nz, num_features=num_features, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)
    LL_eval, log_qz_given_xy_eval, log_pz_eval, log_px_given_zy_eval = latent_gaussian_x_gaussian(z_eval, z_mu_eval, z_log_var_eval, x_mu_eval, x_log_var_eval, sym_x, latent_size=nz, num_features=num_features, eq_samples=sym_eq_samples, iw_samples=sym_iw_samples)

cost_gen = -LL_train
cost = cost_gen + alpha*cost_cla

# count parameters
if distribution == 'bernoulli':
    params = lasagne.layers.get_all_params([classifier, l_dec_x_mu], trainable=True)
    for p in params:
        print p, p.get_value().shape
    params_count = lasagne.layers.count_params([classifier,l_dec_x_mu], trainable=True)
elif distribution == 'gaussian':
    params = lasagne.layers.get_all_params([classifier,l_dec_x_mu, l_dec_x_log_var], trainable=True)
    for p in params:
        print p, p.get_value().shape
    params_count = lasagne.layers.count_params([classifier,l_dec_x_mu, l_dec_x_log_var], trainable=True)
print 'Number of parameters:', params_count

# functions
grads = T.grad(cost, params)
# mgrads = lasagne.updates.total_norm_constraint(grads,max_norm=max_norm)
# cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
updates = lasagne.updates.adam(grads, params, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=sym_lr)

if preprocesses_dataset is not None:
    train_model = theano.function([sym_index_l, sym_index_u, sym_batch_size_l, sym_batch_size_u, sym_lr, sym_eq_samples, sym_iw_samples], [LL_train, log_qz_given_xy_train, log_pz_train, log_px_given_zy_train, classifier_cost_train, accurracy_train_labeled], givens={sym_x_cla:T.concatenate([sh_x_train_labelled[batch_slice_l],sh_x_train_unlabelled[batch_slice_u]], axis=0), sym_x: T.concatenate([sh_x_train_labelled_preprocessed[batch_slice_l],sh_x_train_unlabelled_preprocessed[batch_slice_u]], axis=0), sym_y:sh_t_train_labelled[batch_slice_l]}, updates=updates)
    test_model = theano.function([sym_index, sym_batch_size, sym_eq_samples, sym_iw_samples], [LL_eval, log_qz_given_xy_eval, log_pz_eval, log_px_given_zy_eval, classifier_cost_eval, accurracy_eval], givens={sym_x_cla: sh_x_test[batch_slice], sym_x: sh_x_test_preprocessed[batch_slice], sym_y: sh_t_test[batch_slice]})
else:
    train_model = theano.function([sym_index_l, sym_index_u, sym_batch_size_l, sym_batch_size_u, sym_lr, sym_eq_samples, sym_iw_samples], [LL_train, log_qz_given_xy_train, log_pz_train, log_px_given_zy_train, classifier_cost_train,accurracy_train_labeled], givens={sym_x_cla:T.concatenate([sh_x_train_labelled[batch_slice_l],sh_x_train_unlabelled[batch_slice_u]], axis=0), sym_x: T.concatenate([sh_x_train_labelled[batch_slice_l],sh_x_train_unlabelled[batch_slice_u]], axis=0), sym_y: sh_t_train_labelled[batch_slice_l]}, updates=updates)
    test_model = theano.function([sym_index, sym_batch_size, sym_eq_samples, sym_iw_samples], [LL_eval, log_qz_given_xy_eval, log_pz_eval, log_px_given_zy_eval, classifier_cost_eval, accurracy_eval], givens={sym_x_cla: sh_x_test[batch_slice], sym_x: sh_x_test[batch_slice], sym_y: sh_t_test[batch_slice]})

# random generation for visualization
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
srng_ran = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
srng_ran_share = theano.tensor.shared_randomstreams.RandomStreams(1234)
sym_ran_y = T.ivector('ran_y')

ran_z = T.tile(srng_ran.normal((num_classes,nz)), (num_classes, 1))
if distribution == 'bernoulli':
    random_x_mean = lasagne.layers.get_output(l_dec_x_mu, {l_z:ran_z, l_in_y:sym_ran_y}, deterministic=True)
    random_x = srng_ran_share.binomial(n=1, p=random_x_mean, dtype=theano.config.floatX)
elif distribution == 'gaussian':
    random_x_mean, random_x_log_var = lasagne.layers.get_output([l_dec_x_mu, l_dec_x_log_var], {l_z:ran_z, l_in_y:sym_ran_y}, deterministic=True)
    random_x = srng_ran_share.normal(avg=random_x_mean, std=T.exp(0.5*random_x_log_var))
generate = theano.function(inputs=[sym_ran_y], outputs=[random_x_mean, random_x])


'''
run
'''
# Training and Testing functions
def train_epoch(lr, eq_samples, iw_samples, batch_size):
    costs,log_qz_given_xy,log_pz,log_px_given_zy, loss, accurracy, accurracy_labeled = [],[],[],[],[],[],[]
    n_train_batches_labelled = x_labelled.shape[0] / num_labelled_per_batch
    n_train_batches_unlabelled = x_unlabelled.shape[0] / (batch_size - num_labelled_per_batch)

    for i in range(n_train_batches_unlabelled):
        costs_batch, log_qz_given_xy_batch,log_pz_batch,log_px_given_zy_batch, loss_batch, accurracy_labeled_batch = train_model(i % n_train_batches_labelled, i, num_labelled_per_batch, batch_size-num_labelled_per_batch, lr, eq_samples, iw_samples)
        costs += [costs_batch]
        log_qz_given_xy += [log_qz_given_xy_batch]
        log_pz += [log_pz_batch]
        log_px_given_zy += [log_px_given_zy_batch]
        loss += [loss_batch]
        accurracy_labeled += [accurracy_labeled_batch]
    return np.mean(costs), np.mean(log_qz_given_xy), np.mean(log_pz), np.mean(log_px_given_zy), np.mean(loss), np.mean(accurracy_labeled)

def test_epoch(eq_samples, iw_samples, batch_size):
    n_test_batches = test_x.shape[0] / batch_size
    costs,log_qz_given_xy,log_pz,log_px_given_zy,loss, accurracy = [],[],[],[],[],[]
    for i in range(n_test_batches):
        costs_batch, log_qz_given_xy_batch,log_pz_batch,log_px_given_zy_batch, loss_batch, accurracy_batch = test_model(i, batch_size, eq_samples, iw_samples)
        costs += [costs_batch]
        log_qz_given_xy += [log_qz_given_xy_batch]
        log_pz += [log_pz_batch]
        log_px_given_zy += [log_px_given_zy_batch]
        loss += [loss_batch]
        accurracy += [accurracy_batch]
    return np.mean(costs), np.mean(log_qz_given_xy), np.mean(log_pz), np.mean(log_px_given_zy), np.mean(loss), np.mean(accurracy)


print "Training"

# TRAIN LOOP
LL_train, log_qz_given_x_train, log_pz_train, log_px_given_z_train, loss_train, acc_labeled_train = [],[],[],[],[],[]
LL_test, log_qz_given_x_test, log_pz_test, log_px_given_z_test, loss_test, acc_test = [],[],[],[],[],[]

for epoch in range(1, 1+num_epochs):
    start = time.time()

    # randomly permute data and labels
    p_l = np.random.permutation(x_labelled.shape[0]) 
    sh_x_train_labelled.set_value(x_labelled[p_l])
    sh_t_train_labelled.set_value((y_labelled[p_l]))
    p_u = np.random.permutation(x_unlabelled.shape[0]) 
    sh_x_train_unlabelled.set_value(x_unlabelled[p_u])
    if preprocesses_dataset is not None:
        sh_x_train_labelled_preprocessed.set_value(preprocesses_dataset(x_labelled[p_l]))
        sh_x_train_unlabelled_preprocessed.set_value(preprocesses_dataset(x_unlabelled[p_u]))

    train_out = train_epoch(lr, eq_samples, iw_samples, batch_size)

    if np.isnan(train_out[0]):
        ValueError("NAN in train LL!")

    if epoch >= anneal_lr_epoch and epoch % every_anneal == 0:
        #annealing learning rate
        lr = lr*anneal_lr_factor

    if epoch % eval_epoch == 0:
        t = time.time() - start
        LL_train += [train_out[0]]
        log_qz_given_x_train += [train_out[1]]
        log_pz_train += [train_out[2]]
        log_px_given_z_train += [train_out[3]]
        loss_train +=[train_out[4]]
        acc_labeled_train += [train_out[5]]

        print "calculating LL eq=1, iw=1"
        test_out = test_epoch(eq_samples, iw_samples, batch_size=500)
        LL_test += [test_out[0]]
        log_qz_given_x_test += [test_out[1]]
        log_pz_test += [test_out[2]]
        log_px_given_z_test += [test_out[3]]
        loss_test += [test_out[4]]
        acc_test += [test_out[5]]


        line = "*Epoch=%d\tTime=%.2f\tLR=%.5f\n" %(epoch, t, lr) + \
               "  TRAIN:\tGen_loss=%.5f\tlogq(z|x)=%.5f\tlogp(z)=%.5f\tlogp(x|z)=%.5f\tdis_loss=%.5f\tlabel_error=%.5f\n" %(LL_train[-1], log_qz_given_x_train[-1], log_pz_train[-1], log_px_given_z_train[-1], loss_train[-1], 1-acc_labeled_train[-1]) + \
               "  EVAL-L1:\tGen_loss=%.5f\tlogq(z|x)=%.5f\tlogp(z)=%.5f\tlogp(x|z)=%.5f\tdis_loss=%.5f\terror=%.5f\n" %(LL_test[-1], log_qz_given_x_test[-1], log_pz_test[-1], log_px_given_z_test[-1], loss_test[-1], 1-acc_test[-1])
        print line
        with open(logfile,'a') as f:
            f.write(line + "\n")

    # random generation for visualization
    if epoch % vis_epoch == 0:
        tail='-'+str(epoch)+'.png'
        ran_y = np.int32(np.repeat(np.arange(num_classes), num_classes))
        _x_mean, _x = generate(ran_y)
        _x_mean = _x_mean.reshape((num_generation,-1))
        _x = _x.reshape((num_generation,-1))
        image = paramgraphics.mat_to_img(_x_mean.T, dim_input, colorImg=colorImg, scale=generation_scale, 
            save_path=os.path.join(res_out, 'mean'+tail))

    #save model
    model_out = os.path.join(res_out, 'model')
    if epoch % (vis_epoch*10) == 0:
        if distribution == 'bernoulli':
            all_params=lasagne.layers.get_all_params([classifier, l_dec_x_mu])
        elif distribution == 'gaussian':
            all_params=lasagne.layers.get_all_params([classifier, l_dec_x_mu, l_dec_x_log_var])
        f = gzip.open(model_out + 'epoch%i'%(epoch), 'wb')
        cPickle.dump(all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()