# Max-margin Deep Conditional Generative Models for Semi-Supervised Learning
## [Chongxuan Li](https://github.com/zhenxuan00), Jun Zhu and Bo Zhang

Full [paper](https://arxiv.org/abs/1611.07119), a journal version of our NIPS15 paper (original [paper](https://arxiv.org/abs/1504.06787) and [code](https://github.com/zhenxuan00/mmdgm)). A novel class-condional variants of mmDGMs is proposed.

## Summary of Max-margin Deep Conditional Generative Models (mmDCGMs)

- We boost the effectiveness and efficiency of DGMs in semi-supervised learning by
  - Employing advanced CNNs as the x2y, xy2z and zy2x networks
  - Approximating the posterior inference of labels
  - Proposing powerful max-margin discriminative losses for labeled and unlabeled data
- and the arrived mmDCGMs can
  - Perform efficient inference: constant time with respect to the number of classes
  - Achieve state-of-the-art classification results on sevarl benchmarks: MNIST, SVHN and NORB with 1000 labels and MNIST with full labels
  - Disentangle classes and styles on raw images without preprocessing like PCA given small amount of labels

## Some libs we used in our experiments
> Python
> Numpy
> Scipy
> [Theano](https://github.com/Theano/Theano)
> [Lasagne](https://github.com/Lasagne/Lasagne)
> [Parmesan](https://github.com/casperkaae/parmesan)

## State-of-the-art results on MNIST, SVHN and NORB datasets with 1000 labels and excellent results competitive to best CNNS given all labels on MNIST

> chmod +x *.sh

> ./cdgm-svhn-ssl_1000.sh gpu0 (Run .sh files to obtain corresponding results)

> For small norb dataset, please download the raw images in .MAT format from [http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/) and run datasets_norb.convert_orig_to_np() to convert it into numpy format. 

> See Table 6 and Table 7 in the paper for the classfication results.

## Class conditional generation of raw images given a few labels

### Results on MNIST given 100 labels (left: 100 labeled data sorted by class, right: samples, where each row shares same class and each column shares same style.)
<img src="https://github.com/thu-ml/mmdcgm-ssl/blob/master/images/ssl-mnist-data.png" width="320">  <img src="https://github.com/thu-ml/mmdcgm-ssl/blob/master/images/ssl-mnist-sample.png" width="320">


### Results on SVHN given 1000 labels
<img src="https://github.com/thu-ml/mmdcgm-ssl/blob/master/images/ssl-svhn-data.png" width="320">  <img src="https://github.com/thu-ml/mmdcgm-ssl/blob/master/images/ssl-svhn-sample.png" width="320">

### Results on small NORB given 1000 labels
<img src="https://github.com/thu-ml/mmdcgm-ssl/blob/master/images/ssl-norb-data.png" width="240">  <img src="https://github.com/thu-ml/mmdcgm-ssl/blob/master/images/ssl-norb-sample.png" width="240">
