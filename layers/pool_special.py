import lasagne
from lasagne import init
from lasagne import nonlinearities

import theano.tensor as T
import theano
import numpy as np
import theano.tensor.extra_ops as Textra


__all__ = [
    "UnPoolLayer", # upsampling by setting input to the top-left corner
    "RepeatUnPoolLayer", # upsampling by repeating input
    "UnPoolMaskLayer", # upsampling with pooling location
    "MaxPoolLocationLayer", # get the location of max pooling
]

class UnPoolLayer(lasagne.layers.Layer):
    '''
    Layer that upsampling the input
    
    Parameters
    ----------
    incoming: class `Layer` instance
        dim of incoming: B,C,0,1

    factor : tuple of length 2
        upsample factor
    ----------
    '''
    def __init__(self, incoming, factor, **kwargs):
        super(UnPoolLayer, self).__init__(incoming, **kwargs)
        assert len(factor) == 2
        assert len(self.input_shape) == 4
        self.factor = factor
        window = np.zeros(self.factor, dtype=np.float32)
        window[0, 0] = 1
        image_shape = self.input_shape[1:]
        self.mask = theano.shared(np.tile(window.reshape((1,)+self.factor), image_shape))
        self.mask = T.shape_padleft(self.mask,n_ones=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[:2] + (input_shape[2]*self.factor[0], input_shape[3]*self.factor[1])

    def get_output_for(self, input, **kwargs):
        return Textra.repeat(Textra.repeat(input,self.factor[0],axis=2),self.factor[1],axis=3)*self.mask

class RepeatUnPoolLayer(lasagne.layers.Layer):
    '''
    Layer that upsampling the input
        one unit in the input corresponds a square of units in the output
        all values in the region are same as the corresponding value of input
    
    Parameters
    ----------
    incoming: class `Layer` instance
        dim of incoming: B,C,0,1

    factor : tuple of length 2
        upsample factor
    ----------
    '''
    def __init__(self, incoming, factor, **kwargs):
        super(RepeatUnPoolLayer, self).__init__(incoming, **kwargs)
        assert len(factor) == 2
        assert len(self.input_shape) == 4
        self.factor = factor

    def get_output_shape_for(self, input_shape):
        return input_shape[:2] + (input_shape[2]*self.factor[0], input_shape[3]*self.factor[1])

    def get_output_for(self, input, **kwargs):
        return Textra.repeat(Textra.repeat(input,self.factor[0],axis=2),self.factor[1],axis=3)

class UnPoolMaskLayer(lasagne.layers.MergeLayer):
    '''
    Layer that upsampling the input given the pooling location
    
    Parameters
    ----------
    incoming, mask : class `Layer` instances
        dim of incoming: B,C,0,1
        dim of mask: B,C,0*f1,1*f2

    factor : tuple of length 2
        upsample factor
    ----------
    '''
    def __init__(self, incoming, mask, factor, noise_level=0.7, **kwargs):
        super(UnPoolMaskLayer, self).__init__([incoming, mask], **kwargs)
        assert len(factor) == 2
        assert len(self.input_shapes[0]) == 4
        assert len(self.input_shapes[1]) == 4
        assert self.input_shapes[0][2]*factor[0] == self.input_shapes[1][2]
        assert self.input_shapes[0][3]*factor[1] == self.input_shapes[1][3]
        assert noise_level>=0 and noise_level<=1
        self.factor = factor
        self.noise = noise_level

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, input, **kwargs):
        data, mask_max = input
        #return Textra.repeat(Textra.repeat(data, self.factor[0], axis=2), self.factor[1], axis=3) * mask_max
        window = np.zeros(self.factor, dtype=np.float32)
        window[0, 0] = 1
        mask_unpool = np.tile(window.reshape((1,) + self.factor), self.input_shapes[0][1:])
        mask_unpool = T.shape_padleft(mask_unpool, n_ones=1)

        rs = np.random.RandomState(1234)
        rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))
        mask_binomial = rng.binomial(n=1, p=self.noise, size= self.input_shapes[1][1:])
        mask_binomial = T.shape_padleft(T.cast(mask_binomial, dtype='float32'), n_ones=1)

        mask =  mask_binomial * mask_unpool + (1 - mask_binomial) * mask_max
        return Textra.repeat(Textra.repeat(data,self.factor[0],axis=2),self.factor[1],axis=3)*mask

class MaxPoolLocationLayer(lasagne.layers.Layer):
    '''
    Layer that computes the max-pool location

    Parameters
    ----------
    incoming : a class `Layer` instance
        output shape is 4D

    factor : tuple of length 2
        downsample, fixed to (2, 2) so far

    batch_size : tensor iscalar

    References
    ----------
    '''
    def __init__(self, incoming, factor, batch_size, noise_level=0.5, **kwargs):
        super(MaxPoolLocationLayer, self).__init__(incoming, **kwargs)
        assert factor[0] == 2, factor # only for special (2,2) case
        assert factor[1] == 2, factor
        self.factor = factor
        self.batch_size = batch_size
        self.n_channels = self.input_shape[1]
        self.i_s = self.input_shape[-2:]
        self.noise = noise_level

    def get_output_shape_for(self, input_shape):
        return input_shape

    def _get_output_for(self, input):
        assert input.ndim == 3 # only for 3D
        mask = T.zeros_like(input) # size (None, w, h)
        tmp = T.concatenate([T.shape_padright(input[:, ::2, ::2]), 
            T.shape_padright(input[:, ::2, 1::2]), T.shape_padright(input[:, 1::2, ::2]), 
            T.shape_padright(input[:, 1::2, 1::2])], axis=-1)
        index =  tmp.argmax(axis=-1) # size (None, w/2, h/2)
        i_r = 2*(np.tile(np.arange(self.i_s[0]/2), (self.i_s[1]/2,1))).T
        i_r = index/2 + T.shape_padleft(i_r)
        i_c = 2*(np.tile(np.arange(self.i_s[1]/2), (self.i_s[0]/2,1)))
        i_c = index%2 + T.shape_padleft(i_c)
        i_b = T.tile(T.arange(self.batch_size*self.n_channels),(self.i_s[0]/2*self.i_s[1]/2,1)).T
        mask = T.set_subtensor(mask[i_b.flatten(), i_r.flatten(), i_c.flatten()],1)
        return mask

    def get_output_for(self, input, **kwargs):
        assert input.ndim == 4 # only for 4D
        input_3D = input.reshape((self.batch_size*self.n_channels,)+self.i_s)
        mask_max = self._get_output_for(input_3D)
        return mask_max.reshape((self.batch_size,self.n_channels)+self.i_s)

