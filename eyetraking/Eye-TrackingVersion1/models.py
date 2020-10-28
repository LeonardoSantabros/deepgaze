from __future__ import division

import tensorflow as tf  #New version
from tensorflow.keras.layers import Lambda, concatenate #New version
from tensorflow.keras.layers import Conv2D  #New version
import tensorflow.keras.backend as K

#from keras.layers import Lambda, merge
#from keras.layers import Convolution2D, AtrousConvolution2D, Conv2D
#import theano.tensor as T
#from dcn_vgg import dcn_vgg
import numpy as np

from dcn_resnet import dcn_resnet
from gaussian_prior import LearningPrior
from attentive_convlstm import AttentiveConvLSTM
from config import *

##//////////////////////////////////////////////////////////////////

##FUNCION AGREGADA PARA IMPLEMENTAR CHANNELS FIRST EN CPU'S PARA CONV2D

def Conv2D_NCHW(inputs, 
              filters, 
              kernel_size, 
              strides=(1, 1), 
              padding='valid', 
              dilation_rate=(1, 1), 
              activation=None,
              use_bias=True, 
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros'):
    inputs = K.permute_dimensions(inputs, (0, 2, 3, 1)) 
    outputs = Conv2D(filters=filters, 
                     kernel_size=kernel_size, 
                     strides=strides, 
                     padding=padding,
                     data_format="channels_last", 
                     dilation_rate=dilation_rate, 
                     activation=activation,
                     use_bias=use_bias, 
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer)(inputs) 

    outputs = K.permute_dimensions(outputs, (0, 3, 1, 2)) 
    return outputs

##//////////////////////////////////////////////////////////////////

def repeat(x):
    return K.reshape(K.repeat(K.batch_flatten(x), nb_timestep), (b_s, nb_timestep, 512, shape_r_gt, shape_c_gt)) #(batch, time, channels, height, width)
    #return K.reshape(K.repeat(K.batch_flatten(x), nb_timestep), (b_s, nb_timestep, shape_r_gt, shape_c_gt, 512))  #Nueva version (batch, height, width, channels)

def repeat_shape(s):
    return (s[0], nb_timestep) + s[1:]


def upsampling(x):
    #x = outs
    x = K.permute_dimensions(x, (0, 2, 3, 1)) #AGREGADO. Forma del input requerida [batch, height, width, channels].
    #x = tf.image.resize(x, [upsampling_factor, upsampling_factor])  #New version
    x = tf.image.resize(x, [shape_r_out, shape_c_out])  #New version ?????
    x = K.permute_dimensions(x, (0, 3, 1, 2)) #AGREGADO. Regresamos la forma original.
    return x
    #return T.nnet.abstract_conv.bilinear_upsampling(input=x, ratio=upsampling_factor, num_input_channels=1, batch_size=b_s)


def upsampling_shape(s):
    #Input s = (1, 16, 16, 1) <- Eso cambio
    return s[:2] + (s[2] * upsampling_factor, s[3] * upsampling_factor)


# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    return 10 * K.sum(K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1), axis=-1)


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    sum_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_true /= (sum_y_true + K.epsilon())
    y_pred /= (sum_y_pred + K.epsilon())

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=2), axis=2)
    sum_x = K.sum(K.sum(y_true, axis=2), axis=2)
    sum_y = K.sum(K.sum(y_pred, axis=2), axis=2)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=2), axis=2)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=2), axis=2)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt((sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N))

    return -2 * num / den


# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)), 
                                                                   shape_r_out, axis=-1)), shape_c_out, axis=-1)
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_mean)), 
                                                               shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.expand_dims(y_std)), 
                                                              shape_r_out, axis=-1)), shape_c_out, axis=-1)

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(K.sum(K.sum(y_true * y_pred, axis=2), axis=2) / K.sum(K.sum(y_true, axis=2), axis=2))


# Gaussian priors initialization
def gaussian_priors_init(shape, name=None):
    means = np.random.uniform(low=0.3, high=0.7, size=shape[0] // 2)
    covars = np.random.uniform(low=0.05, high=0.3, size=shape[0] // 2)
    return K.variable(np.concatenate((means, covars), axis=0), name=name)

#APAGAMOS LA VGG16\\\\\\\\\\\\\\\\
# def sam_vgg(x):
#     # Dilated Convolutional Network
#     dcn = dcn_vgg(input_tensor=x[0])

#     # Attentive Convolutional LSTM
#     att_convlstm = Lambda(repeat, repeat_shape)(dcn.output)
#     att_convlstm = AttentiveConvLSTM(nb_filters_in=512, nb_filters_out=512, nb_filters_att=512,
#                                       nb_cols=3, nb_rows=3)(att_convlstm)

#     # Learned Prior (1)
#     priors1 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(x[1])
#     concateneted = merge([att_convlstm, priors1], mode='concat', concat_axis=1)
#     learned_priors1 = AtrousConvolution2D(512, 5, 5, border_mode='same', activation='relu',
#                                           atrous_rate=(4, 4))(concateneted)

#     # Learned Prior (2)
#     priors2 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(x[1])
#     concateneted = merge([learned_priors1, priors2], mode='concat', concat_axis=1)
#     learned_priors2 = AtrousConvolution2D(512, 5, 5, border_mode='same', activation='relu',
#                                           atrous_rate=(4, 4))(concateneted)

#     # Final Convolutional Layer
#     outs = Convolution2D(1, 1, 1, border_mode='same', activation='relu')(learned_priors2)
#     outs_up = Lambda(upsampling, upsampling_shape)(outs)

#     return [outs_up, outs_up, outs_up]


def sam_resnet(x):
    #x = [x, x_maps]
    # Dilated Convolutional Network
    print("Iniciando sam_resnet")
    print("Iniciando dcn_resnet...")
    dcn = dcn_resnet(input_tensor=x[0]) #Ready!!
    aux = K.permute_dimensions(dcn.output, (0, 3, 1, 2))   #Agregado para poner channels_first como en el codigo original.
    #conv_feat = Convolution2D(512, 3, 3, border_mode='same', activation='relu')(dcn.output)
    
    # New Version. Input shape = (None, 2048, 30, 40) output shape=(None, 512, 30, 40)
    #conv_feat = Conv2D(512, 
    #                   (3, 3), 
    #                   padding='same', 
    #                   activation='relu',
    #                   data_format="channels_first")(aux)     # GPU New Version

    conv_feat = Conv2D_NCHW(aux,512, 
                            (3, 3), 
                            padding='same', 
                            activation='relu')     # CPU New Version NCHW

    # Attentive Convolutional LSTM
    print("Iniciando att_convlstm...")
    att_convlstm = Lambda(repeat, repeat_shape)(conv_feat) #Output shape=(1, 4, 512, 30, 40)
    #x = att_convlstm
    #l = AttentiveConvLSTM(nb_filters_in=512, nb_filters_out=512, nb_filters_att=512,nb_cols=3, nb_rows=3)
    #l(x)
    att_convlstm = AttentiveConvLSTM(nb_filters_in=512,  #Output shape=(1, 512, 30, 40)
                                     nb_filters_out=512, 
                                     nb_filters_att=512,
                                     nb_cols=3, 
                                     nb_rows=3)(att_convlstm)
   
    # Learned Prior (1)

    #Input shape = (None, 16, 30, 40), output shape = (1, 16, 30, 40)
    print("Iniciando LearningPrior 1...")
    priors1 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(x[1])

    #concateneted = merge([att_convlstm, priors1], mode='concat', concat_axis=1)
    #print(att_convlstm.shape)
    #print(priors1.shape)
    #attentive = att_convlstm * 1  #Eliminando un bug
    #prior = priors1 * 1           #Eliminando un bug
    #print(attentive.shape)
    #print(prior.shape)
    #concateneted = concatenate([attentive, prior], axis=1)          #Nueva version sin BUG

    print("Concatenando...")
    concateneted = concatenate([att_convlstm, priors1], axis=1)    #Version con BUG
 
    #learned_priors1 = AtrousConvolution2D(512, 5, 5, border_mode='same', activation='relu',
    #                                      atrous_rate=(4, 4))(concateneted)

    #learned_priors1 = Conv2D(512, (5, 5), dilation_rate=(4, 4), activation='relu',
    #           data_format="channels_first", padding='same')(concateneted)  #New version for GPU
        
    learned_priors1 = Conv2D_NCHW(concateneted, 512, 
                                  (5, 5), 
                                  dilation_rate=(4, 4), 
                                  activation='relu', 
                                  padding='same')  #New version for CPU 
    # Learned Prior (2) 
    print("Iniciando LearningPrior 2...")
    priors2 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(x[1])
    
    #learned_priors1 = learned_priors1 * 1   #Eliminando un bug
    #priors2 = priors2 * 1                   #Eliminando un bug
    
    #concateneted = merge([learned_priors1, priors2], mode='concat', concat_axis=1)
    print("Concatenando...")
    concateneted = concatenate([learned_priors1, priors2], axis=1) 

    #learned_priors2 = AtrousConvolution2D(512, 5, 5, border_mode='same', activation='relu',
    #                                      atrous_rate=(4, 4))(concateneted)

    #learned_priors2 = Conv2D(512, (5, 5), dilation_rate=(4, 4), activation='relu',
    #           data_format="channels_first", padding='same')(concateneted)  #New version for GPU
     
    learned_priors2 = Conv2D_NCHW(concateneted, 512, 
                                  (5, 5), 
                                  dilation_rate=(4, 4), 
                                  activation='relu', 
                                  padding='same')  #New version for CPU 
       
    # Final Convolutional Layer
    print("Final Convolutional Layer")
    #outs = Convolution2D(1, 1, 1, border_mode='same', activation='relu')(learned_priors2)
    #outs = Conv2D(1, (1, 1), padding='same', data_format="channels_first", activation='relu')(learned_priors2) #New version for GPU output shape=(1, 1, 30, 40)
    outs = Conv2D_NCHW(learned_priors2, 1, 
                       (1, 1), 
                       padding='same', 
                       activation='relu') #New version for COU output shape=(1, 1, 30, 40)


    #VALIDAR ESTA FUNCION \\\\\\\\\\\\\\\
    outs_up = Lambda(upsampling, upsampling_shape)(outs) # Input shape=(1, 1, 30, 40)
    #print(outs_up.shape) #(1, 1, 480, 640)
    
    print("Finalizado sam_resnet")
    return [outs_up, outs_up, outs_up]   #When passing a list as loss, it should have one entry per model outputs


