import tensorflow as tf
import numpy as np 
from MnasNet import MnasNet
import tensorflow.keras.backend as K 

class AdaInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             center=True,
             scale=True,
             **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
    
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        super(AdaInstanceNormalization, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = tf.keras.backend.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))
        
        gamma = inputs[1]
        beta = inputs[2]
        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = tf.keras.backend.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = tf.keras.backend.std(inputs[0], reduction_axes, keepdims=True) 
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]


class AdaInstanceNormalization_nobias(tf.keras.layers.Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             center=True,
             scale=True,
             **kwargs):
        super(AdaInstanceNormalization_nobias, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
    
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        super(AdaInstanceNormalization_nobias, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = tf.keras.backend.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))
        
        gamma = inputs[1]
        #beta = inputs[2]
        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = tf.keras.backend.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = tf.keras.backend.std(inputs[0], reduction_axes, keepdims=True) 
        normed = (inputs[0] - mean) / stddev

        return normed * gamma
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization_nobias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]




def decoder(use_bias_effect,content_len=2048):
    input_style = tf.keras.layers.Input(shape=(content_len,))
    input_content = tf.keras.layers.Input(shape=(content_len,))


    
    dense = tf.keras.layers.Dense(units=1024*4*4,use_bias=use_bias_effect,kernel_initializer = 'he_normal')(input_content)
    dense = tf.keras.layers.LeakyReLU(alpha=0.3)(dense)
    dense = tf.keras.layers.Reshape((4,4,1024))(dense)

    con_dense = tf.keras.layers.Dense(units=256*4*4,use_bias=use_bias_effect,kernel_initializer = 'he_normal')(input_style)
    con_dense = tf.keras.layers.LeakyReLU(alpha=0.3)(con_dense)
    con_dense = tf.keras.layers.Reshape((4,4,256))(con_dense)

    up6 = tf.keras.layers.Conv2D(1024, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(dense))
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.3)(up6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    con6 = tf.keras.layers.Conv2D(256, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(con_dense))
    con6 = tf.keras.layers.LeakyReLU(alpha=0.3)(con6)
    con6 = tf.keras.layers.BatchNormalization()(con6)
    con_6 = tf.keras.layers.Concatenate()([conv6,con6])
    conv6 = tf.keras.layers.Conv2D(1024, 3, padding = 'same',use_bias=use_bias_effect, kernel_initializer = 'he_normal')(con_6)
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv6)

    up7 = tf.keras.layers.Conv2D(512, 2, padding = 'same',use_bias=use_bias_effect, kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    conv7 = tf.keras.layers.LeakyReLU(alpha=0.3)(up7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    con7 = tf.keras.layers.Conv2D(128, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(con6))
    con7 = tf.keras.layers.LeakyReLU(alpha=0.3)(con7)
    con7 = tf.keras.layers.BatchNormalization()(con7)
    con_7 = tf.keras.layers.Concatenate()([conv7,con7])

    conv7 = tf.keras.layers.Conv2D(512, 3, use_bias=use_bias_effect,padding = 'same', kernel_initializer = 'he_normal')(con_7)
    conv7 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv7)



    up8 = tf.keras.layers.Conv2D(256, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    conv8 = tf.keras.layers.LeakyReLU(alpha=0.3)(up8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8)
    con8 = tf.keras.layers.Conv2D(128, 3, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(con7))
    con8 = tf.keras.layers.LeakyReLU(alpha=0.3)(con8)
    con8 = tf.keras.layers.BatchNormalization()(con8)
    con_8 = tf.keras.layers.Concatenate()([conv8,con8])
    conv8 = tf.keras.layers.Conv2D(256, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(con_8)
    conv8 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv8)



    up9 = tf.keras.layers.Conv2D(128, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv8))
    conv9 = tf.keras.layers.LeakyReLU(alpha=0.3)(up9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9)
    con9 = tf.keras.layers.Conv2D(128, 3, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(con8))
    con9 = tf.keras.layers.LeakyReLU(alpha=0.3)(con9)
    con9 = tf.keras.layers.BatchNormalization()(con9)
    con_9 = tf.keras.layers.Concatenate()([conv9,con9])
    conv9 = tf.keras.layers.Conv2D(128, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(con_9)
    conv9 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv9)



    up10 = tf.keras.layers.Conv2D(64, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv9))
    conv10 = tf.keras.layers.LeakyReLU(alpha=0.3)(up10)
    conv10 = tf.keras.layers.BatchNormalization()(conv10)
    con10 = tf.keras.layers.Conv2D(64, 3, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(con9))
    con10 = tf.keras.layers.LeakyReLU(alpha=0.3)(con10)
    con10 = tf.keras.layers.BatchNormalization()(con10)
    con_10 = tf.keras.layers.Concatenate()([conv10,con10])
    conv10 = tf.keras.layers.Conv2D(128, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(con_10)
    conv10 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv10)


    up11 = tf.keras.layers.Conv2D(32, 2, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv10))
    conv11 = tf.keras.layers.LeakyReLU(alpha=0.3)(up11)
    conv11 = tf.keras.layers.Conv2D(32, 3, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(conv11)
    conv11 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv11)


    conv11 = tf.keras.layers.Conv2D(3, 3, padding = 'same', use_bias=use_bias_effect,kernel_initializer = 'he_normal')(conv11)
    conv11 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv11)




    #conv10 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv10)


    return tf.keras.models.Model(inputs = [input_content,input_style], outputs = [conv11])

def decoder_full(content_len=2048):



    def kl_loss(args):
        z_mean, z_log_var = args
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return kl_loss


    input_style = tf.keras.layers.Input(shape=(content_len,))
    input_content = tf.keras.layers.Input(shape=(content_len,))
    #input_content_style = tf.keras.layers.Input(shape=(content_len,))


#FCN
    # gamma_style = tf.keras.layers.Dense(units=content_len,use_bias=False)(input_style)
    # gamma_style = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style)
    # gamma_style = tf.keras.layers.Dense(units=content_len/2,use_bias=False)(gamma_style)
    # gamma_style = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style)
    # gamma_style = tf.keras.layers.Dense(units=content_len/2,use_bias=False)(gamma_style)
    # gamma_style = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style)
    # gamma_style = tf.keras.layers.Dense(units=content_len/2,use_bias=False)(gamma_style)
    # gamma_style = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style)

    gamma_content = tf.keras.layers.Dense(units=content_len,use_bias=False)(input_content)
    gamma_content = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_content)
    gamma_content = tf.keras.layers.Dense(units=content_len/2,use_bias=False)(gamma_content)
    gamma_content = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_content)

    gamma_content1_1 = tf.keras.layers.Dense(units=128,use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(gamma_content)
    gamma_content1_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_content1_1)
    gamma_content1_2 = tf.keras.layers.Dense(units=128,use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(gamma_content1_1)
    gamma_content1_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_content1_2)

    beta_content1_1 = tf.keras.layers.Dense(units=128,use_bias=False)(gamma_content)
    beta_content1_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_content1_1)
    beta_content1_2 = tf.keras.layers.Dense(units=128,use_bias=False)(beta_content1_1)
    beta_content1_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_content1_2)

    gamma_5_1 = tf.keras.layers.Reshape((1,1,128))(gamma_content1_1)
    gamma_5_2 = tf.keras.layers.Reshape((1,1,128))(gamma_content1_2)
    beta_5_1 = tf.keras.layers.Reshape((1,1,128))(beta_content1_1)
    beta_5_2 = tf.keras.layers.Reshape((1,1,128))(beta_content1_2)


    # gamma_style1_1 = tf.keras.layers.Dense(units=1024,use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.01, max_value=0.05, rate=1.0, axis=0)
    # ,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(gamma_style)
    # gamma_style1_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style1_1)
    # gamma_style1_2 = tf.keras.layers.Dense(units=1024,use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.01, max_value=0.05, rate=1.0, axis=0)
    # ,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(gamma_style1_1)
    # gamma_style1_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style1_2)   

    gamma_style2_1 = tf.keras.layers.Dense(units=512,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    gamma_style2_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style2_1)
    #gamma_style2_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(gamma_style2_1)

    gamma_style2_2 = tf.keras.layers.Dense(units=512,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    gamma_style2_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style2_2)
    #gamma_style2_2 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(gamma_style2_2)

    gamma_style3_1 = tf.keras.layers.Dense(units=256,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    gamma_style3_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style3_1)
    #gamma_style3_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(gamma_style3_1)
    
    gamma_style3_2 = tf.keras.layers.Dense(units=256,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    gamma_style3_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style3_2)
    #gamma_style3_2 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(gamma_style3_2)

    gamma_style4_1 = tf.keras.layers.Dense(units=256,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    gamma_style4_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style4_1)
    #gamma_style4_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(gamma_style4_1)

    gamma_style4_2 = tf.keras.layers.Dense(units=256,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    gamma_style4_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style4_2)
    #gamma_style4_2 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(gamma_style4_2)

    # gamma_style5_1 = tf.keras.layers.Dense(units=128,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(gamma_style)
    # gamma_style5_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style5_1)    
    # gamma_style5_2 = tf.keras.layers.Dense(units=128,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(gamma_style)
    # gamma_style5_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(gamma_style5_2)
    

    # beta_style1_1 = tf.keras.layers.Dense(units=1024,use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.5, max_value=1, rate=1.0, axis=0)
    # ,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(gamma_style)
    # beta_style1_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style1_1)
    # beta_style1_2 = tf.keras.layers.Dense(units=1024,use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.5, max_value=1, rate=1.0, axis=0)
    # ,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(beta_style1_1)
    # beta_style1_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style1_2)

    beta_style2_1 = tf.keras.layers.Dense(units=512,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    beta_style2_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style2_1)

    beta_style2_2 = tf.keras.layers.Dense(units=512,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    beta_style2_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style2_2)

    beta_style3_1 = tf.keras.layers.Dense(units=256,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    beta_style3_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style3_1)

    beta_style3_2 = tf.keras.layers.Dense(units=256,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    beta_style3_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style3_2)

    beta_style4_1 = tf.keras.layers.Dense(units=256,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    beta_style4_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style4_1)

    beta_style4_2 = tf.keras.layers.Dense(units=256,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003)
    ,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=0.5, rate=1.0, axis=0))(input_style)
    beta_style4_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style4_2)

    # beta_style5_1 = tf.keras.layers.Dense(units=128,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(gamma_style)
    # beta_style5_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style5_1)
    # beta_style5_1 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(beta_style5_1)

    # beta_style5_2 = tf.keras.layers.Dense(units=128,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(gamma_style)
    # beta_style5_2 = tf.keras.layers.LeakyReLU(alpha=0.3)(beta_style5_2)
    # beta_style5_2 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(beta_style5_2)

    #gamma_1_1 = tf.keras.layers.Reshape((1,1,1024))(gamma_style1_1)
    gamma_2_1 = tf.keras.layers.Reshape((1,1,512))(gamma_style2_1)
    gamma_3_1 = tf.keras.layers.Reshape((1,1,256))(gamma_style3_1)
    gamma_4_1 = tf.keras.layers.Reshape((1,1,256))(gamma_style4_1)
    #gamma_1_2 = tf.keras.layers.Reshape((1,1,1024))(gamma_style1_2)
    gamma_2_2 = tf.keras.layers.Reshape((1,1,512))(gamma_style2_2)
    gamma_3_2 = tf.keras.layers.Reshape((1,1,256))(gamma_style3_2)
    gamma_4_2 = tf.keras.layers.Reshape((1,1,256))(gamma_style4_2)
    # gamma_5_1 = tf.keras.layers.Reshape((1,1,128))(gamma_style5_1)
    # gamma_5_2 = tf.keras.layers.Reshape((1,1,128))(gamma_style5_2)

    #beta_1_1 = tf.keras.layers.Reshape((1,1,1024))(beta_style1_1)
    beta_2_1 = tf.keras.layers.Reshape((1,1,512))(beta_style2_1)
    beta_3_1 = tf.keras.layers.Reshape((1,1,256))(beta_style3_1)
    beta_4_1 = tf.keras.layers.Reshape((1,1,256))(beta_style4_1)
    #beta_1_2 = tf.keras.layers.Reshape((1,1,1024))(beta_style1_2)
    beta_2_2 = tf.keras.layers.Reshape((1,1,512))(beta_style2_2)
    beta_3_2 = tf.keras.layers.Reshape((1,1,256))(beta_style3_2)
    beta_4_2 = tf.keras.layers.Reshape((1,1,256))(beta_style4_2)
    # beta_5_1 = tf.keras.layers.Reshape((1,1,128))(beta_style5_1)
    # beta_5_2 = tf.keras.layers.Reshape((1,1,128))(beta_style5_2)

    dense = tf.keras.layers.Dense(units=1024*4*4,use_bias=False)(input_content)
    dense = tf.keras.layers.LeakyReLU(alpha=0.3)(dense)
    dense = tf.keras.layers.Reshape((4,4,1024))(dense)   
    up6 = tf.keras.layers.Conv2DTranspose(1024,4,(2,2),padding='same',use_bias=False,kernel_initializer='he_normal')(dense)
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.3)(up6)
    # up6 = tf.keras.layers.Conv2D(1024, 2, padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(up6)
    # conv6 = tf.keras.layers.LeakyReLU(alpha=0.3)(up6)
    #conv6 = AdaInstanceNormalization()([up6,gamma_1,beta_1])
    #conv6 = tf.keras.layers.LeakyReLU(alpha=0.3)(up6)
    conv6 = tf.keras.layers.Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal',use_bias=False)(conv6)
    conv6 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv6)

    up5 = tf.keras.layers.Conv2DTranspose(1024,4,(2,2),padding='same',use_bias=False,kernel_initializer='he_normal')(conv6)
    conv5 = tf.keras.layers.LeakyReLU(alpha=0.3)(up5)

    conv5 = tf.keras.layers.Conv2D(1024, 3, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(conv5)
    conv5 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv5)


    up_7 = tf.keras.layers.Conv2DTranspose(512,4,(2,2),padding='same',use_bias=False,kernel_initializer='he_normal')(conv5)
    conv7 = tf.keras.layers.LeakyReLU(alpha=0.3)(up_7)
    # conv_7 = tf.keras.layers.Conv2D(512, 3, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(conv7)
    # conv_7 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv_7)
    conv7 = AdaInstanceNormalization()([conv7,gamma_2_1,beta_2_1])
    #conv7 = AdaInstanceNormalization_nobias()([conv7,gamma_2_1])
    conv_7 = tf.keras.layers.Conv2D(512, 3, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(conv7)
    conv_7 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv_7)
    conv7 = AdaInstanceNormalization()([conv7,gamma_2_2,beta_2_2])
    #conv7 = AdaInstanceNormalization_nobias()([conv7,gamma_2_2])


    up8 = tf.keras.layers.Conv2DTranspose(256,4,(2,2),padding='same',use_bias=False,kernel_initializer='he_normal')(conv_7)
    conv8 = tf.keras.layers.LeakyReLU(alpha=0.3)(up8)
    # up8 = tf.keras.layers.Conv2D(256, 2, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(up8)
    # conv8 = tf.keras.layers.LeakyReLU(alpha=0.3)(up8)
    conv8 = AdaInstanceNormalization()([conv8,gamma_3_1,beta_3_1])
    #conv8 = AdaInstanceNormalization_nobias()([conv8,gamma_3_1])
    conv8 = tf.keras.layers.Conv2D(256, 3, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(conv8)
    conv8 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv8)
    conv8 = AdaInstanceNormalization()([conv8,gamma_3_2,beta_3_2])
    #conv8 = AdaInstanceNormalization_nobias()([conv8,gamma_3_2])

    up9 = tf.keras.layers.Conv2DTranspose(256,4,(2,2),padding='same',use_bias=False,kernel_initializer='he_normal')(conv8)
    conv9 = tf.keras.layers.LeakyReLU(alpha=0.3)(up9)
    # up9 = tf.keras.layers.Conv2D(128, 2, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(up9)
    # conv9 = tf.keras.layers.LeakyReLU(alpha=0.3)(up9)
    conv9 = AdaInstanceNormalization()([conv9,gamma_4_1,beta_4_1])
    #conv9 = AdaInstanceNormalization_nobias()([conv9,gamma_4_1])
    conv9 = tf.keras.layers.Conv2D(256, 3, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(conv9)
    conv9 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv9)
    conv9 = AdaInstanceNormalization()([conv9,gamma_4_2,beta_4_2])
    #conv9 = AdaInstanceNormalization_nobias()([conv9,gamma_4_2])

    up10 = tf.keras.layers.Conv2DTranspose(128,4,(2,2),padding='same',use_bias=False,kernel_initializer='he_normal')(conv9)
    conv10 = tf.keras.layers.LeakyReLU(alpha=0.3)(up10)
    # up10 = tf.keras.layers.Conv2D(64, 2, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(up10)
    # conv10 = tf.keras.layers.LeakyReLU(alpha=0.3)(up10)
    conv10 = AdaInstanceNormalization()([conv10,gamma_5_1,beta_5_1])
    conv10 = tf.keras.layers.Conv2D(128, 3, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(conv10)
    conv10 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv10)
    conv10 = AdaInstanceNormalization()([conv10,gamma_5_2,beta_5_2])

    conv10 = tf.keras.layers.Conv2D(3, 3, padding = 'same', use_bias=False,kernel_initializer = 'he_normal')(conv10)
    conv10 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv10)



    z_loss_2_1 = tf.keras.layers.Lambda(kl_loss)([beta_style2_1, gamma_style2_1])
    z_loss_2_2 = tf.keras.layers.Lambda(kl_loss)([beta_style2_2, gamma_style2_2])
    z_loss_3_1 = tf.keras.layers.Lambda(kl_loss)([beta_style3_1, gamma_style3_1])
    z_loss_3_2 = tf.keras.layers.Lambda(kl_loss)([beta_style3_2, gamma_style3_2])
    z_loss_4_1 = tf.keras.layers.Lambda(kl_loss)([beta_style4_1, gamma_style4_1])
    z_loss_4_2 = tf.keras.layers.Lambda(kl_loss)([beta_style4_2, gamma_style4_2])


    model1 = tf.keras.models.Model(inputs = [input_content,input_style], outputs = [conv10,z_loss_2_1,z_loss_2_2,z_loss_3_1,z_loss_3_2,z_loss_4_1,z_loss_4_2])

    return model1








def reconstruction_model(content_len=2048):

    def sampling(args): 
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(content_len,), mean=0.,
                                stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def kl_loss(args):
        z_mean, z_log_var = args
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return kl_loss

    input_face = tf.keras.layers.Input(shape=(256,256,3))
    style_face = tf.keras.layers.Input(shape=(256,256,3))
    model_face_style = MnasNet(n_classes=content_len,FC=True)
    model_struct = MnasNet(n_classes=content_len)
    #model_face_content = MnasNet(n_classes=1024) 
    #structure_model = decoder_mini()
    recon_model = decoder_full(content_len=content_len)

    style_vec = model_face_style(style_face)
    content_vec = model_struct(input_face)

    # z_mean = tf.keras.layers.Dense(units=content_len,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(style_vec)
    # z_mean_norm = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(z_mean)
    # z_log_var = tf.keras.layers.Dense(units=content_len,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(style_vec)
    # z_log_var_norm = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999)(z_log_var)

    # z = tf.keras.layers.Lambda(sampling)([z_mean_norm, z_log_var_norm])
    # z_loss = tf.keras.layers.Lambda(kl_loss)([z_mean_norm, z_log_var_norm])
    
    #content_style_vec = tf.keras.layers.Dense(units=content_len,use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(l=0.003))(content_vec)
    #content_vec,content_style_vec = tf.keras.layers.Lambda(tf.split, arguments={'axis': -1, 'num_or_size_splits': 2})(content_vec)
    #face_mini = structure_model(input_1)
    #res = tf.keras.layers.Dot(axes=1,normalize=True)([style_vec,content_vec])


    #content_vec = tf.keras.layers.Dense(units=content_len,use_bias=False)(content_vec)
    face,kl_2_1,kl_2_2,kl_3_1,kl_3_2,kl_4_1,kl_4_2 = recon_model([content_vec,style_vec])
    #face = recon_model([content_vec,z])
    re_model = tf.keras.models.Model(inputs=[style_face,input_face],outputs=[face,style_vec,kl_2_1,kl_2_2,kl_3_1,kl_3_2,kl_4_1,kl_4_2])
    style_model = tf.keras.models.Model(inputs=style_face,outputs=[style_vec])
    content_model = tf.keras.models.Model(inputs=input_face,outputs=[content_vec])

#xiugai
    return re_model,style_model,content_model,recon_model






    


def voice_pic(pic_shape):
    inputs_voice = tf.keras.layers.Input(shape=(64,13))
    inputs_pic = tf.keras.layers.Input(shape=pic_shape) 
    
    conv1 = tf.keras.layers.Conv3D(filters=32, kernel_size=[2, 3, 3], strides=1, name='conv1_1',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2,3]),
                                   padding='same')(inputs_pic)
    conv1 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv1)
    conv1 = tf.keras.layers.Conv3D(filters=32, kernel_size=[2, 5, 5], strides=1, name='conv1_2',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2,3]),
                                   padding='same')(conv1)
    conv1 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv1)
    pool1 = tf.keras.layers.MaxPool3D(pool_size=[1, 3, 3], padding='same', strides=[1, 2, 2], name='pool1')(conv1)


    conv2 = tf.keras.layers.Conv3D(filters=64, kernel_size=[2, 3, 3], strides=1, name='conv2_1',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2,3]),
                                   padding='same')(pool1)
    conv2 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv2)
    conv2 = tf.keras.layers.Conv3D(filters=64, kernel_size=[2, 5, 5], strides=1, name='conv2_2',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2,3]),
                                   padding='same')(conv2)
    conv2 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv2)
                                   
                                   
    pool2 = tf.keras.layers.MaxPool3D(pool_size=[1, 3, 3], padding='same', strides=[1, 2, 2], name='pool2')(conv2)

    conv3 = tf.keras.layers.Conv3D(filters=128, kernel_size=[2, 3, 3], strides=1, name='conv3_1',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2,3]),
                                   padding='same')(pool2)
    conv3 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv3)
    conv3 = tf.keras.layers.Conv3D(filters=128, kernel_size=[2, 3, 3], strides=1, name='conv3_2',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2,3]),
                                   padding='same')(conv3)
    conv3 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv3)
    pool3 = tf.keras.layers.MaxPool3D(pool_size=[1, 3, 3], padding='same', strides=[1, 2, 2], name='pool3')(conv3)
    
    lstm_pic = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=[3, 3], strides=1, padding='same', name='lstm1',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2,3]),
                                       activation=tf.keras.activations.tanh, return_sequences=False)(pool3)
                                       
    lstm_voice = tf.keras.layers.LSTM(units=64, name='lstm2',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2,3]),
                                       activation=tf.keras.activations.tanh, return_sequences=False)(inputs_voice)
    
    voice = tf.keras.layers.Dense(units=64,use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=0))(lstm_voice)
    mix = AdaInstanceNormalization()([voice,lstm_pic])
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=[0,1,2]))(mix)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=256,use_bias=False,kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1, max_value=100, rate=1.0, axis=0))(x)
    return tf.keras.models.Model(inputs=[inputs_voice,inputs_pic],outputs=x)





if __name__ == '__main__':
    model,_,_,_ = reconstruction_model()
    #model.summary()
    #another = tf.keras.models.Model(inputs=model.input,outputs=model.get_layer('lambda').output)
    #model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
