'''
This code is part of the Keras ResNet-50 model
'''
from __future__ import print_function
from __future__ import absolute_import

from tensorflow.keras.layers import add, Input, Activation              #Nueva version
from tensorflow.keras.layers import MaxPooling2D, ZeroPadding2D, Conv2D #Nueva version
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras.utils.data_utils import get_file

import os

#from keras.layers import merge, Input, Activation
#from keras.layers.convolutional import AtrousConvolution2D
    
#TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
TH_WEIGHTS_PATH_NO_TOP = os.getcwd() + "\\weights\\resnet50_weights_th_dim_ordering_th_kernels_notop.h5"

from keras.engine import saving
import h5py

##///////////////////////////////////////////////////////////
#Clase nueva agregada al codigo para agregar un nuevo metodo

class ModelAux(Model):     #Agregado
    @saving.allow_read_from_gcs
    def load_weights_new(self, filepath, by_name=False,
                     skip_mismatch=False, reshape=False):
        """Loads all layer weights from a HDF5 save file.

        If `by_name` is False (default) weights are loaded
        based on the network's topology, meaning the architecture
        should be the same as when the weights were saved.
        Note that layers that don't have weights are not taken
        into account in the topological ordering, so adding or
        removing layers is fine as long as they don't have weights.

        If `by_name` is True, weights are loaded into layers
        only if they share the same name. This is useful
        for fine-tuning or transfer-learning models where
        some of the layers have changed.

        # Arguments
            filepath: String, path to the weights file to load.
            by_name: Boolean, whether to load weights by name
                or by topological order.
            skip_mismatch: Boolean, whether to skip loading of layers
                where there is a mismatch in the number of weights,
                or a mismatch in the shape of the weight
                (only valid when `by_name`=True).
            reshape: Reshape weights to fit the layer when the correct number
                of weight arrays is present but their shape does not match.


        # Raises
            ImportError: If h5py is not available.
        """

        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            if by_name:
                saving.load_weights_from_hdf5_group_by_name(
                    f, self.layers, skip_mismatch=skip_mismatch,
                    reshape=reshape)
            else:
                saving.load_weights_from_hdf5_group(
                    f, self.layers, reshape=reshape)
            if hasattr(f, 'close'):
                f.close()
            elif hasattr(f.file, 'close'):
                f.file.close()
                
##//////////////////////////////////////////////////////////////////


def identity_block(input_tensor, kernel_size, filters, stage, block):
    nb_filter1, nb_filter2, nb_filter3 = filters
    #bn_axis = 1
    bn_axis = -1            #Nueva version
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), data_format="channels_last", 
               name=conv_name_base + '2a')(input_tensor)                   #Nueva version
    #x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), data_format="channels_last", 
               padding='same', name=conv_name_base + '2b')(x)              #Nueva version
    #x = Convolution2D(nb_filter2, kernel_size, kernel_size,
    #                  border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), data_format="channels_last", 
               name=conv_name_base + '2c')(x)           #Nueva version
    #x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])                          #Nueva version
    #x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    #bn_axis = 1
    bn_axis = -1            #Nueva version
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, data_format="channels_last", 
               name=conv_name_base + '2a')(input_tensor)         #Nueva version
    #x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
    #                  name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), data_format="channels_last",
               padding='same', name=conv_name_base + '2b')(x)     #Nueva version
    #x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
    #                  name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), data_format="channels_last", name=conv_name_base + '2c')(x)   #Nueva version
    #x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, data_format="channels_last", 
               name=conv_name_base + '1')(input_tensor)           #Nueva version  
    #shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
    #                         name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])                   #Nueva version
    #x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block_atrous(input_tensor, kernel_size, filters, stage, block, atrous_rate=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    #bn_axis = 1
    bn_axis = -1            #Nueva version

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), data_format="channels_last", 
               name=conv_name_base + '2a')(input_tensor) #New version
    #x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate, data_format="channels_last",
                            padding='same', name=conv_name_base + '2b')(x)              #New version
    #x = AtrousConvolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
    #                        atrous_rate=atrous_rate, name=conv_name_base + '2b')(x)    #Deprecated
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), data_format="channels_last", 
               name=conv_name_base + '2c')(x) #New version
    #x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), data_format="channels_last", 
                      name=conv_name_base + '1')(input_tensor)
    #shortcut = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])                   #Nueva version
    #x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def identity_block_atrous(input_tensor, kernel_size, filters, stage, block, atrous_rate=(2, 2)):
    nb_filter1, nb_filter2, nb_filter3 = filters
    #bn_axis = 1
    bn_axis = -1            #Nueva version

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), data_format="channels_last", name=conv_name_base + '2a')(input_tensor) #New version
    #x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate, data_format="channels_last",
                            padding='same', name=conv_name_base + '2b')(x)                  #Nueva version
    #x = AtrousConvolution2D(nb_filter2, kernel_size, kernel_size, atrous_rate=atrous_rate,
    #                        border_mode='same', name=conv_name_base + '2b')(x)             #Deprecated
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), data_format="channels_last", name=conv_name_base + '2c')(x) #New version
    #x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])                   #Nueva version
    #x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def dcn_resnet(input_tensor=None):    
    ''''Functional API method: you start from `Input`,you chain layer calls to 
    specify the model's forward pass,and finally you create your model from inputs
    and outputs'''
    #input_shape = (3, None, None)
    input_shape = (None, None, 3)                   #Nueva version
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):     #Si lo que entra no es un tensor creamos la input layer.
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    #bn_axis = 1
    bn_axis = -1            #Nueva version
    
    # conv_1
    x = ZeroPadding2D((3, 3))(img_input)            #Input shape (height, width, channels)
    #x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), data_format="channels_last", name='conv1')(x)     #New version
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x) #New version
    
    # conv_2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # conv_3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # conv_4
    x = conv_block_atrous(x, 3, [256, 256, 1024], stage=4, block='a', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='b', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='c', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='d', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='e', atrous_rate=(2, 2))
    x = identity_block_atrous(x, 3, [256, 256, 1024], stage=4, block='f', atrous_rate=(2, 2))

    # conv_5
    x = conv_block_atrous(x, 3, [512, 512, 2048], stage=5, block='a', atrous_rate=(4, 4))
    x = identity_block_atrous(x, 3, [512, 512, 2048], stage=5, block='b', atrous_rate=(4, 4))
    x = identity_block_atrous(x, 3, [512, 512, 2048], stage=5, block='c', atrous_rate=(4, 4))

    # Create model
    #model = Model(inputs=img_input, outputs=x)  #Functional API method
    model = ModelAux(inputs=img_input, outputs=x)  #New Version
    
    # Load weights
    weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5', TH_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models', md5_hash='f64f049c92468c9affcd44b0976cdafe',
                            cache_dir=os.getcwd() + "\\weights")
    # print('____________________________________________')
    # import numpy as np 
    # nro = 2                  
    # print(model.layers[nro])
    # weight = model.layers[nro].get_weights()
    # if(len(weight) == 0):
    #     print("La layer no tiene pesos") 
    # else:          
    #     weight0 = np.array(weight[0])
    #     print(weight0.shape)
    # print('____________________________________________')
    
    #model.load_weights(weights_path)
    model.load_weights_new(weights_path,reshape = True)  #Nuevo metodo agregado

    print("Red convolucional dcn_resnet cargada")
    return model