'''
objectives
'''
import numpy as np
import theano.tensor as T
import theano
import lasagne

from parmesan.distributions import log_stdnormal, log_normal2, log_bernoulli

def multiclass_s3vm_loss(predictions, targets, num_labelled, weight_decay, norm_type=2, form ='mean_class', alpha_hinge=1., alpha_hat=1., alpha_reg=1., alpha_decay=1., delta=1., entropy_term=False):
    '''
    predictions: 
        size L x nc
             U x nc
    targets: 
        size L x nc

    output:
        weighted sum of hinge loss, hat loss, balance constraint and weight decay
    '''
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')

    hinge_loss = multiclass_hinge_loss_(predictions[:num_labelled], targets, delta)
    hat_loss = multiclass_hat_loss(predictions[num_labelled:], delta)
    regularization = balance_constraint(predictions, targets, num_labelled, norm_type, form)
    if not entropy_term:
        return alpha_hinge*hinge_loss.mean() + alpha_hat*hat_loss.mean() + alpha_reg*regularization + alpha_decay*weight_decay
    else:
        # given an unlabeled data, when treat hat loss as the entropy term derived from a lowerbound, it should conflict to current prediction, which is quite strange but true ... the entropy term enforce the discriminator to predict unlabeled data uniformly as a regularization
        # max entropy regularization provides a tighter lowerbound but hurt the semi-supervised learning performance as it conflicts to the hat loss ...
        return alpha_hinge*hinge_loss.mean() - alpha_hat*hat_loss.mean() + alpha_reg*regularization + alpha_decay*weight_decay

def multiclass_hinge_loss_(predictions, targets, delta=1):
    return lasagne.objectives.multiclass_hinge_loss(predictions, targets, delta)

def multiclass_hinge_loss(predictions, targets, weight_decay, alpha_decay=1., delta=1):
    return multiclass_hinge_loss_(predictions, targets, delta).mean() + alpha_decay*weight_decay

def multiclass_hat_loss(predictions, delta=1):
    targets = T.argmax(predictions, axis=1)
    return multiclass_hinge_loss(predictions, targets, delta)

def balance_constraint(predictions, targets, num_labelled, norm_type=2, form='mean_class'):
    '''
    balance constraint
    ------
    norm_type: type of norm 
            l2 or l1
    form: form of regularization
            mean_class: average mean activation of u and l data should be the same over each class
            mean_all: average mean activation of u and l data should be the same over all data
            ratio: 

    '''
    p_l = predictions[:num_labelled]
    p_u = predictions[num_labelled:]
    t_l = targets
    t_u = T.argmax(p_u, axis=1)
    num_cls = predictions.shape[1]
    t_u = theano.tensor.extra_ops.to_one_hot(t_u, num_cls)
    if form == 'mean_class':
        res = (p_l*t_l).mean(axis=0) - (p_u*t_u).mean(axis=0)
    elif form == 'mean_all':
        res = p_l.mean(axis=0) - p_u.mean(axis=0)
    elif form == 'ratio':
        pass

    # res should be a vector with length number_class
    return res.norm(norm_type)

def latent_gaussian_x_gaussian(z, z_mu, z_log_var, x_mu, x_log_var, x, latent_size, num_features, eq_samples, iw_samples, epsilon=1e-6):
    # reshape the variables so batch_size, eq_samples and iw_samples are separate dimensions
    z = z.reshape((-1, eq_samples, iw_samples, latent_size))
    x_mu = x_mu.reshape((-1, eq_samples, iw_samples, num_features))
    x_log_var = x_log_var.reshape((-1, eq_samples, iw_samples, num_features))

    # dimshuffle x, z_mu and z_log_var since we need to broadcast them when calculating the pdfs
    x = x.reshape((-1,num_features))
    x = x.dimshuffle(0, 'x', 'x', 1)                    # size: (batch_size, eq_samples, iw_samples, num_features)
    z_mu = z_mu.dimshuffle(0, 'x', 'x', 1)              # size: (batch_size, eq_samples, iw_samples, num_latent)
    z_log_var = z_log_var.dimshuffle(0, 'x', 'x', 1)    # size: (batch_size, eq_samples, iw_samples, num_latent)

    # calculate LL components, note that the log_xyz() functions return log prob. for indepenedent components separately 
    # so we sum over feature/latent dimensions for multivariate pdfs
    log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis=3)
    log_pz = log_stdnormal(z).sum(axis=3)
    #log_px_given_z = log_bernoulli(x, T.clip(x_mu, epsilon, 1 - epsilon)).sum(axis=3)
    log_px_given_z = log_normal2(x, x_mu, x_log_var).sum(axis=3)

    #all log_*** should have dimension (batch_size, eq_samples, iw_samples)
    # Calculate the LL using log-sum-exp to avoid underflow
    a = log_pz + log_px_given_z - log_qz_given_x    # size: (batch_size, eq_samples, iw_samples)
    a_max = T.max(a, axis=2, keepdims=True)         # size: (batch_size, eq_samples, 1)

    LL = T.mean(a_max) + T.mean( T.log( T.mean(T.exp(a-a_max), axis=2) ) )

    return LL, T.mean(log_qz_given_x), T.mean(log_pz), T.mean(log_px_given_z)

def latent_gaussian_x_bernoulli(z, z_mu, z_log_var, x_mu, x, latent_size, num_features, eq_samples, iw_samples, epsilon=1e-6):
    """
    Latent z       : gaussian with standard normal prior
    decoder output : bernoulli

    When the output is bernoulli then the output from the decoder
    should be sigmoid. The sizes of the inputs are
    z: (batch_size*eq_samples*iw_samples, num_latent)
    z_mu: (batch_size, num_latent)
    z_log_var: (batch_size, num_latent)
    x_mu: (batch_size*eq_samples*iw_samples, num_features)
    x: (batch_size, num_features)

    Reference: Burda et al. 2015 "Importance Weighted Autoencoders"
    """

    # reshape the variables so batch_size, eq_samples and iw_samples are separate dimensions
    z = z.reshape((-1, eq_samples, iw_samples, latent_size))
    x_mu = x_mu.reshape((-1, eq_samples, iw_samples, num_features))

    # dimshuffle x, z_mu and z_log_var since we need to broadcast them when calculating the pdfs
    x = x.reshape((-1,num_features))
    x = x.dimshuffle(0, 'x', 'x', 1)                    # size: (batch_size, eq_samples, iw_samples, num_features)
    z_mu = z_mu.dimshuffle(0, 'x', 'x', 1)              # size: (batch_size, eq_samples, iw_samples, num_latent)
    z_log_var = z_log_var.dimshuffle(0, 'x', 'x', 1)    # size: (batch_size, eq_samples, iw_samples, num_latent)

    # calculate LL components, note that the log_xyz() functions return log prob. for indepenedent components separately 
    # so we sum over feature/latent dimensions for multivariate pdfs
    log_qz_given_x = log_normal2(z, z_mu, z_log_var).sum(axis=3)
    log_pz = log_stdnormal(z).sum(axis=3)
    log_px_given_z = log_bernoulli(x, T.clip(x_mu, epsilon, 1 - epsilon)).sum(axis=3)

    #all log_*** should have dimension (batch_size, eq_samples, iw_samples)
    # Calculate the LL using log-sum-exp to avoid underflow
    a = log_pz + log_px_given_z - log_qz_given_x    # size: (batch_size, eq_samples, iw_samples)
    a_max = T.max(a, axis=2, keepdims=True)         # size: (batch_size, eq_samples, 1)

    # LL is calculated using Eq (8) in Burda et al.
    # Working from inside out of the calculation below:
    # T.exp(a-a_max): (batch_size, eq_samples, iw_samples)
    # -> subtract a_max to avoid overflow. a_max is specific for  each set of
    # importance samples and is broadcasted over the last dimension.
    #
    # T.log( T.mean(T.exp(a-a_max), axis=2) ): (batch_size, eq_samples)
    # -> This is the log of the sum over the importance weighted samples
    #
    # The outer T.mean() computes the mean over eq_samples and batch_size
    #
    # Lastly we add T.mean(a_max) to correct for the log-sum-exp trick
    LL = T.mean(a_max) + T.mean( T.log( T.mean(T.exp(a-a_max), axis=2) ) )

    return LL, T.mean(log_qz_given_x), T.mean(log_pz), T.mean(log_px_given_z)
