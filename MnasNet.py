from tensorflow.keras import optimizers, layers, models, callbacks, utils, preprocessing, regularizers,initializers,constraints
from tensorflow.keras  import backend as K
import tensorflow as tf
import numpy as np

class InstanceNormalization(layers.Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
 
    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')
 
        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')
 
        self.input_spec = layers.InputSpec(ndim=ndim)
 
        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)
 
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True
 
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))
 
        if self.axis is not None:
            del reduction_axes[self.axis]
 
        del reduction_axes[0]
 
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev
 
        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]
 
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed
 
    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def MnasNet(n_classes=512, input_shape=(256, 256, 3), alpha=1,FC=False):
	inputs = layers.Input(shape=input_shape)

	x = conv_bn(inputs, 32*alpha, 3,   strides=2)
	x = sepConv_bn_noskip(x, 16*alpha, 3,  strides=1) 
	# MBConv3 3x3
	x = MBConv_idskip(x, filters=24, kernel_size=3,  strides=2, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, filters=24, kernel_size=3,  strides=1, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, filters=24, kernel_size=3,  strides=1, filters_multiplier=3, alpha=alpha)
	# MBConv3 5x5
	x = MBConv_idskip(x, filters=40, kernel_size=5,  strides=2, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, filters=40, kernel_size=5,  strides=1, filters_multiplier=3, alpha=alpha)
	x = MBConv_idskip(x, filters=40, kernel_size=5,  strides=1, filters_multiplier=3, alpha=alpha)
	# MBConv6 5x5
	x = MBConv_idskip(x, filters=80, kernel_size=5,  strides=2, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=80, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	out1 = MBConv_idskip(x, filters=80, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 3x3
	x = MBConv_idskip(out1, filters=96, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)
	out2 = MBConv_idskip(x, filters=96, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)
	# MBConv6 5x5
	x = MBConv_idskip(out2, filters=192, kernel_size=5,  strides=2, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
	x = MBConv_idskip(x, filters=192, kernel_size=5,  strides=1, filters_multiplier=6, alpha=alpha)
    # MBConv6 3x3
	x = MBConv_idskip(x, filters=320, kernel_size=3,  strides=1, filters_multiplier=6, alpha=alpha)

	# FC + POOL
	x = conv_bn(x, filters=1152*alpha, kernel_size=1,   strides=1)
	x = layers.GlobalAveragePooling2D()(x)
	predictions = layers.Dense(n_classes,use_bias=False,kernel_regularizer=regularizers.l2(l=0.003),
	kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.3, max_value=1, rate=1.0, axis=0))(x)
	gamma_style = tf.keras.layers.Dense(units=n_classes,use_bias=False,kernel_regularizer=regularizers.l2(l=0.0003))(predictions)

		#gamma_style = tf.keras.layers.Lambda(tf.contrib.layers.layer_norm, arguments={'axis': -1, 'num_or_size_splits': 2})(gamma_style)
		#gamma_style = InstanceNormalization(epsilon=1e-4)(gamma_style)
	


	if FC:
		gamma_style = tf.keras.layers.BatchNormalization(epsilon=1e-3)(gamma_style)
		return models.Model(inputs=inputs, outputs=gamma_style)
	else :
		return models.Model(inputs=inputs, outputs=gamma_style)

# Convolution with batch normalization
def conv_bn(x, filters, kernel_size,  strides=1, alpha=1, activation=True):
	"""Convolution Block
	This function defines a 2D convolution operation with BN and relu6.
	# Arguments
		x: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		alpha: An integer which multiplies the filters dimensionality
		activation: A boolean which indicates whether to have an activation after the normalization 
	# Returns
		Output tensor.
	"""
	filters = _make_divisible(filters * alpha)
	x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
									use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))(x)
	x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)  
	if activation:
		x = layers.LeakyReLU(alpha=0.2)(x)
	return x

# Depth-wise Separable Convolution with batch normalization 
def depthwiseConv_bn(x, depth_multiplier, kernel_size,  strides=1):
	""" Depthwise convolution 
	The DepthwiseConv2D is just the first step of the Depthwise Separable convolution (without the pointwise step).
	Depthwise Separable convolutions consists in performing just the first step in a depthwise spatial convolution 
	(which acts on each input channel separately).
	
	This function defines a 2D Depthwise separable convolution operation with BN and relu6.
	# Arguments
		x: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
	# Returns
		Output tensor.
	"""

	x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=depth_multiplier,
									padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l=0.0003))(x)  
	x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(x)  
	x = layers.LeakyReLU(alpha=0.2)(x)
	return x

def sepConv_bn_noskip(x, filters, kernel_size,  strides=1):
	""" Separable convolution block (Block F of MNasNet paper https://arxiv.org/pdf/1807.11626.pdf)
	
	# Arguments
		x: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
	# Returns
		Output tensor.
	"""

	x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
	x = conv_bn(x, filters=filters, kernel_size=1, strides=1)

	return x

# Inverted bottleneck block with identity skip connection
def MBConv_idskip(x_input, filters, kernel_size,  strides=1, filters_multiplier=1, alpha=1):
	""" Mobile inverted bottleneck convolution (Block b, c, d, e of MNasNet paper https://arxiv.org/pdf/1807.11626.pdf)
	
	# Arguments
		x: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel_size: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
		alpha: An integer which multiplies the filters dimensionality
	# Returns
		Output tensor.
	"""

	depthwise_conv_filters = _make_divisible(x_input.shape[3].value) 
	pointwise_conv_filters = _make_divisible(filters * alpha)

	x = conv_bn(x_input, filters=depthwise_conv_filters * filters_multiplier, kernel_size=1, strides=1)
	x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
	x = conv_bn(x, filters=pointwise_conv_filters, kernel_size=1, strides=1, activation=False)

	# Residual connection if possible
	if strides==1 and x.shape[3] == x_input.shape[3]:
		return  layers.add([x_input, x])
	else: 
		return x


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor=8, min_value=None):
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v



if __name__ == "__main__":

	model = MnasNet()
	model.compile(optimizer='adam')
	model.summary()