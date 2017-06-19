import numpy as np
import pickle as pkl
import cPickle as cPkl
import gzip
import tarfile
import fnmatch
import os
import urllib
from scipy.io import loadmat

def _unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

def _get_datafolder_path():
    #full_path = os.path.abspath('.')
    #path = full_path +'/data'
    path = '/home/chongxuan/mfs/data'
    return path

def _download_svhn(datasets_dir=_get_datafolder_path()+'/svhn/'):    
    url = 'http://ufldl.stanford.edu/housenumbers/'
    data_file_list = ['train_32x32.mat', 'test_32x32.mat', 'extra_32x32.mat']

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    for data_file in data_file_list:
        if not os.path.isfile(os.path.join(datasets_dir,data_file)):
            urllib.urlretrieve(os.path.join(url,data_file), data_file)
    
    batch1_data = []
    batch1_labels = []
    batch2_data = []
    batch2_labels = []
    from random import shuffle

    train = loadmat(os.path.join(datasets_dir,data_file_list[0]))
    x = train['X'].transpose((2, 0, 1, 3)).reshape((3072, -1))
    y = train['y'].reshape((-1,))
    for i in np.arange(len(y)):
        if y[i] == 10:
            y[i] = 0
    index = np.arange(len(y))
    shuffle(index)
    x = x[:, index]
    y = y[index]

    count = np.zeros((10,), 'int32')
    for i in np.arange(len(y)):
        if count[y[i]] < 400:
            count[y[i]] += 1
            batch2_data.append(x[:, i])
            batch2_labels.append(y[i])
        else:
            batch1_data.append(x[:, i])
            batch1_labels.append(y[i])

    print '---train'
    extra = loadmat(os.path.join(datasets_dir,data_file_list[2]))
    x = extra['X'].transpose((2, 0, 1, 3)).reshape((3072, -1))
    y = extra['y'].reshape((-1,))
    del extra
    for i in np.arange(len(y)):
        if y[i] == 10:
            y[i] = 0
    index = np.arange(len(y))
    shuffle(index)
    x = x[:, index]
    y = y[index]

    count = np.zeros((10,), 'int32')
    for i in np.arange(len(y)):
        if count[y[i]] < 200:
            count[y[i]] += 1
            batch2_data.append(x[:, i])
            batch2_labels.append(y[i])
        else:
            batch1_data.append(x[:, i])
            batch1_labels.append(y[i])
    batch1_data = np.asarray(batch1_data)
    batch2_data = np.asarray(batch2_data)
    batch1_labels = np.asarray(batch1_labels)
    batch2_labels = np.asarray(batch2_labels)
    del x, y

    print '---extra'

    test = loadmat(os.path.join(datasets_dir,data_file_list[1]))
    x = test['X'].transpose((2, 0, 1, 3)).reshape((3072, -1))
    y = test['y'].reshape((-1,))
    for i in np.arange(len(y)):
        if y[i] == 10:
            y[i] = 0
    batch3_data = x
    batch3_labels = []
    for i in np.arange(len(y)):
        batch3_labels.append(y[i])
    batch3_data = np.asarray(batch3_data).T
    batch3_labels = np.asarray(batch3_labels)

    print 'Check n x f'
    print batch1_data.shape
    print batch1_labels.shape
    print batch2_data.shape
    print batch2_labels.shape
    print batch3_data.shape
    print batch3_labels.shape

    f = file(datasets_dir+"/svhn.bin","wb")
    np.save(f,batch1_data)
    np.save(f,batch1_labels)
    np.save(f,batch2_data)
    np.save(f,batch2_labels)
    np.save(f,batch3_data)
    np.save(f,batch3_labels)
    f.close()

def load_svhn(datasets_dir=_get_datafolder_path()+'/svhn/', normalized=True, centered=True):
    data_file = os.path.join(datasets_dir, 'svhn.bin')

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    if not os.path.isfile(data_file):
        _download_svhn()
    
    f = file(data_file,"rb")
    train_x = np.load(f)
    train_y = np.load(f)
    valid_x = np.load(f)
    valid_y = np.load(f)
    test_x = np.load(f)
    test_y = np.load(f)
    f.close()
    if normalized:
        train_x = train_x/256.0
        valid_x = valid_x/256.0
        test_x = test_x/256.0

    avg = None
    if centered:
        avg = train_x.mean(axis=0,keepdims=True)
        train_x = train_x - avg
        test_x = test_x - avg
        valid_x = valid_x - avg
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y, avg

def load_cifar10(datasets_dir=_get_datafolder_path()+'/cifar10', num_val=None, normalized=True, centered=True):
    # this code is largely cp from Kyle Kastner:
    #
    # https://gist.github.com/kastnerkyle/f3f67424adda343fef40
    
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    data_file = os.path.join(datasets_dir, 'cifar-10-python.tar.gz')
    data_dir = os.path.join(datasets_dir, 'cifar-10-batches-py')

    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)

    if not os.path.isfile(data_file):
        urllib.urlretrieve(url, data_file)
        org_dir = os.getcwd()
        with tarfile.open(data_file) as tar:
            os.chdir(datasets_dir)
            tar.extractall()
        os.chdir(org_dir)

    train_files = []
    for filepath in fnmatch.filter(os.listdir(data_dir), 'data*'):
        train_files.append(os.path.join(data_dir, filepath))
    train_files = sorted(train_files, key=lambda x: x.split("_")[-1])

    test_file = os.path.join(data_dir, 'test_batch')

    x_train, targets_train = [], []
    for f in train_files:
        d = _unpickle(f)
        x_train.append(d['data'])
        targets_train.append(d['labels'])
    x_train = np.array(x_train, dtype='uint8')
    shp = x_train.shape
    x_train = x_train.reshape(shp[0] * shp[1], 3, 32, 32)
    targets_train = np.array(targets_train)
    targets_train = targets_train.ravel()

    d = _unpickle(test_file)
    x_test = d['data']
    targets_test = d['labels']
    x_test = np.array(x_test, dtype='uint8')
    x_test = x_test.reshape(-1, 3, 32, 32)
    targets_test = np.array(targets_test)
    targets_test = targets_test.ravel()
    
    if normalized:
        x_train = x_train/256.0
        x_test = x_test/256.0
    if centered:
        avg = x_train.mean(axis=0,keepdims=True)
        x_train = x_train - avg
        x_test = x_test - avg

    if num_val is not None:
        perm = np.random.permutation(x_train.shape[0])
        x = x_train[perm]
        y = targets_train[perm]

        x_valid = x[:num_val]
        targets_valid = y[:num_val]
        x_train = x[num_val:]
        targets_train = y[num_val:]
        return (x_train, targets_train,
                x_valid, targets_valid,
                x_test, targets_test)
    else:
        return x_train, targets_train, x_test, targets_test