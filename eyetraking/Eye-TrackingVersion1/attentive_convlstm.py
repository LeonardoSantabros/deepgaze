#Definimos la capa AttentiveConvLSTM. 

from __future__ import division

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv2D #New version
from tensorflow.keras import initializers, activations
import tensorflow as tf

##//////////////////////////////////////////////////////////////////
'''https://stackoverflow.com/questions/55442053/keras-custom-layer-not-returning-weights-unlike-normal-layer'''
##CLASE AGREGADA PARA IMPLEMENTAR CHANNELS FIRST EN CPU'S PARA CONV2D
#FALTA AGREGARLE EL add_weight() al metodo build()

class Conv2D_NCHW_C(Layer):
    '''We have the calls to add_weight(), and then call the super's build()'''
    def __init__(self,
                 filters, 
                 kernel_size, 
                 strides=(1, 1), 
                 padding='valid', 
                 dilation_rate=(1, 1), 
                 activation=None,
                 use_bias=None, 
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size  
        self.strides = strides 
        self.padding = padding         
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.outputs = None

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3 , 3, int(input_shape[1]), self.filters),
                                      initializer = self.kernel_initializer,
                                      trainable=True)
        
        if (self.use_bias == True): 
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer = self.bias_initializer,
                                        trainable=True)
        
        super(Conv2D_NCHW_C, self).build(input_shape)
      
    def call(self, inputs):
        inputs = K.permute_dimensions(inputs, (0, 2, 3, 1)) 
        #print(self.outputs)
        #Evitamos que se creen variables multiples veces.
        if self.outputs is None:  
            self.outputs = Conv2D(filters=self.filters, 
                 kernel_size=self.kernel_size, 
                 strides=self.strides, 
                 padding=self.padding,
                 data_format="channels_last", 
                 dilation_rate=self.dilation_rate, 
                 activation=self.activation,
                 use_bias=self.use_bias, 
                 kernel_initializer=self.kernel_initializer,
                 bias_initializer=self.bias_initializer)(inputs) 
            self.outputs = K.permute_dimensions(self.outputs, (0, 3, 1, 2)) 
        return self.outputs

##//////////////////////////////////////////////////////////////////

class AttentiveConvLSTM(Layer):
    def __init__(self, nb_filters_in, nb_filters_out, nb_filters_att, nb_rows, nb_cols,
                 init='normal', inner_init='orthogonal', attentive_init='zero',
                 activation='tanh', inner_activation='sigmoid',
                 W_regularizer=None, U_regularizer=None,
                 weights=None, go_backwards=False,
                 **kwargs):
        self.nb_filters_in = nb_filters_in
        self.nb_filters_out = nb_filters_out
        self.nb_filters_att = nb_filters_att
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.init = initializers.get(init)
        self.inner_init = initializers.get(inner_init)
        self.attentive_init = initializers.get(attentive_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.initial_weights = weights
        self.go_backwards = go_backwards

        self.W_regularizer = W_regularizer
        self.U_regularizer = U_regularizer
        self.input_spec = [InputSpec(ndim=5)]

        #super(AttentiveConvLSTM, self).__init__(**kwargs) #Python2
        super().__init__(**kwargs)   #Nueva version python3
        
    def get_output_shape_for(self, input_shape):
        return input_shape[:1] + (self.nb_filters_out,) + input_shape[3:]

    def compute_mask(self, input, mask):
        return None

    def get_initial_states(self, x):
        '''Fuente: https://stackoverflow.com/questions/55925283/what-is-the-difference-between-conv2d-and-conv2d-in-keras'''
        initial_state = K.sum(x, axis=1)                # initial_state shape=(1, 512, 30, 40)
        #print("---------------get_initial_states 1")
        #initial_state = K.conv2d(initial_state, 
        #                         K.zeros((self.nb_filters_out, self.nb_filters_in, 1, 1)), 
        #                         border_mode='same')
        #print(initial_state.shape)  #Last before the fail (1, 512, 30, 40)
        #Apagado forzoso
        #initial_state = K.conv2d(initial_state, #ESTA FALLANDO
        #                         K.zeros((1,1,self.nb_filters_out, self.nb_filters_in)), # (kernel_size, kernel_size, in_channels, out_channels)
        #                         padding='same',data_format='channels_first') 
        #print(initial_state.shape)
        #print("---------------get_initial_states 2")
        initial_states = [initial_state for _ in range(len(self.states))]

        return initial_states #Retorna lista de tensores shape=(1, 512, 30, 40)

    def build(self, input_shape):
        #print("===============")
        #print(input_shape)
        #print("===============")
        self.input_spec = [InputSpec(shape=input_shape)] #input_shape = (None, 4, 512, 30, 40)
        self.states = [None, None]
        #self.trainable_weights = [] 
        self.trainable_weights2 = []    #New version      

        self.W_a = Conv2D_NCHW_C(self.nb_filters_att, 
                          (self.nb_rows, self.nb_cols),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer = self.init)      #New vesion
        self.U_a = Conv2D_NCHW_C(self.nb_filters_att, 
                          (self.nb_rows, self.nb_cols),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer = self.init)      #New vesion
        self.V_a = Conv2D_NCHW_C(1, 
                          (self.nb_rows, self.nb_cols), 
                          padding='same',
                          use_bias=False, #ESTO NO ESTA FUNCIONANDO
                          kernel_initializer = self.attentive_init)      #New vesion
        #self.W_a = Convolution2D(self.nb_filters_att, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
        #self.U_a = Convolution2D(self.nb_filters_att, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
        #self.V_a = Convolution2D(1, self.nb_rows, self.nb_cols, border_mode='same', bias=False, init=self.attentive_init)
       
        #self.W_a.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_att))       #New vesion [None,30,40,512]
        #self.U_a.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_in))       #New vesion
        #self.V_a.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_att))       #New vesion
        self.W_a.build((input_shape[0], self.nb_filters_att, input_shape[3], input_shape[4]))      #[1,512,30,40]
        self.U_a.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
        self.V_a.build((input_shape[0], self.nb_filters_att, input_shape[3], input_shape[4]))

        self.W_a.built = True
        self.U_a.built = True
        self.V_a.built = True

        self.W_i = Conv2D_NCHW_C(self.nb_filters_out,
                          (self.nb_rows, self.nb_cols),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer = self.init)    #New vesion
        self.U_i = Conv2D_NCHW_C(self.nb_filters_out,
                          (self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer = self.inner_init)    #New vesion
        #self.W_i = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
        #self.U_i = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.inner_init)

        #self.W_i.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_in))       #New vesion
        #self.U_i.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_out))       #New vesion
        self.W_i.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
        self.U_i.build((input_shape[0], self.nb_filters_out, input_shape[3], input_shape[4]))

        self.W_i.built = True
        self.U_i.built = True

        self.W_f = Conv2D_NCHW_C(self.nb_filters_out,
                          (self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True, 
                          kernel_initializer = self.init)    #New vesion
        self.U_f = Conv2D_NCHW_C(self.nb_filters_out,
                          (self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer = self.inner_init)    #New vesion
        #self.W_f = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
        #self.U_f = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.inner_init)

        #self.W_f.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_in))       #New vesion
        #self.U_f.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_out))       #New vesion
        self.W_f.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
        self.U_f.build((input_shape[0], self.nb_filters_out, input_shape[3], input_shape[4]))
        
        self.W_f.built = True
        self.U_f.built = True

        self.W_c = Conv2D_NCHW_C(self.nb_filters_out,
                          (self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer = self.init)    #New vesion
        self.U_c = Conv2D_NCHW_C(self.nb_filters_out,
                          (self.nb_rows, self.nb_cols),
                          padding='same',
                          use_bias=True,
                          kernel_initializer = self.inner_init)    #New vesion
        #self.W_c = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
        #self.U_c = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.inner_init)

        #self.W_c.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_in))       #New vesion
        #self.U_c.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_out))       #New vesion
        self.W_c.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
        self.U_c.build((input_shape[0], self.nb_filters_out, input_shape[3], input_shape[4]))
        
        self.W_c.built = True
        self.U_c.built = True

        self.W_o = Conv2D_NCHW_C(self.nb_filters_out, 
                          (self.nb_rows, self.nb_cols), 
                          padding='same', 
                          use_bias=True, 
                          kernel_initializer = self.init)    #New vesion
        self.U_o = Conv2D_NCHW_C(self.nb_filters_out, 
                          (self.nb_rows, self.nb_cols), 
                          padding='same', 
                          use_bias=True, 
                          kernel_initializer = self.inner_init)    #New vesion
        #self.W_o = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.init)
        #self.U_o = Convolution2D(self.nb_filters_out, self.nb_rows, self.nb_cols, border_mode='same', bias=True, init=self.inner_init)

        #self.W_o.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_in))       #New vesion
        #self.U_o.build((input_shape[0], input_shape[3], input_shape[4], self.nb_filters_out))       #New vesion
        self.W_o.build((input_shape[0], self.nb_filters_in, input_shape[3], input_shape[4]))
        self.U_o.build((input_shape[0], self.nb_filters_out, input_shape[3], input_shape[4]))
        
        self.W_o.built = True
        self.U_o.built = True

        # self.trainable_weights = []
        # self.trainable_weights.extend(self.W_a.trainable_weights)
        # self.trainable_weights.extend(self.U_a.trainable_weights)
        # self.trainable_weights.extend(self.V_a.trainable_weights)
        # self.trainable_weights.extend(self.W_i.trainable_weights)
        # self.trainable_weights.extend(self.U_i.trainable_weights)
        # self.trainable_weights.extend(self.W_f.trainable_weights)
        # self.trainable_weights.extend(self.U_f.trainable_weights)
        # self.trainable_weights.extend(self.W_c.trainable_weights)
        # self.trainable_weights.extend(self.U_c.trainable_weights)
        # self.trainable_weights.extend(self.W_o.trainable_weights)
        # self.trainable_weights.extend(self.U_o.trainable_weights)

        self.trainable_weight2 = []     #NUEVAS VERSIONES
        self.trainable_weights2.extend(self.W_a.trainable_weights)
        self.trainable_weights2.extend(self.U_a.trainable_weights)
        self.trainable_weights2.extend(self.V_a.trainable_weights)
        self.trainable_weights2.extend(self.W_i.trainable_weights)
        self.trainable_weights2.extend(self.U_i.trainable_weights)
        self.trainable_weights2.extend(self.W_f.trainable_weights)
        self.trainable_weights2.extend(self.U_f.trainable_weights)
        self.trainable_weights2.extend(self.W_c.trainable_weights)
        self.trainable_weights2.extend(self.U_c.trainable_weights)
        self.trainable_weights2.extend(self.W_o.trainable_weights)
        self.trainable_weights2.extend(self.U_o.trainable_weights)

    def preprocess_input(self, x):
        return x

    def step(self, x, states):    
        # x.shape=(1, 512, 30, 40)
        # states : lista de tensores shape=(1, 512, 30, 40)
        h_tm1 = states[0]
        c_tm1 = states[1]
      
        #print("Checkpoint 1--------------")
        e = self.V_a(K.tanh(self.W_a(h_tm1) + self.U_a(x))) #e.shape (1, 1, 30, 40)
        #print("Checkpoint 2--------------")
        a = K.reshape(K.softmax(K.batch_flatten(e)), (x.shape[0], 1, x.shape[2], x.shape[3])) #Nueva version a.shape (1, 1, 30, 40)
        #a = K.reshape(K.softmax(K.batch_flatten(e)), (x_shape[0], 1, x_shape[2], x_shape[3])) 
        #print("Checkpoint 3--------------")
        x_tilde = x * K.repeat_elements(a, x.shape[1], 1) #Nueva version x_tilde.shape=(1, 512, 30, 40)
        #x_tilde = x * K.repeat_elements(a, x_shape[1], 1)
        #print("Checkpoint 4--------------")
        x_i = self.W_i(x_tilde)
        x_f = self.W_f(x_tilde)
        x_c = self.W_c(x_tilde)
        x_o = self.W_o(x_tilde)

        i = self.inner_activation(x_i + self.U_i(h_tm1))
        f = self.inner_activation(x_f + self.U_f(h_tm1))
        c = f * c_tm1 + i * self.activation(x_c + self.U_c(h_tm1))
        o = self.inner_activation(x_o + self.U_o(h_tm1))

        h = o * self.activation(c)
        #print("Dime que llegaste/////////////////////")
        return h, [h, c]

    def get_constants(self, x):
        return []

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape #input_shape = (None, 4, 512, 30, 40)
        initial_states = self.get_initial_states(x) #x.shape = (1, 4, 512, 30, 40) output shape=(1, 512, 30, 40)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x) #output shape=(1, 4, 512, 30, 40)

        #print("______________Llegamos________________")
        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=False,
                                             mask=mask,
                                             constants=constants,
                                             unroll=False,
                                             input_length=input_shape[1])
        
        #print("Estamos en la salida ______________________")
        #print(last_output) # shape=(1, 512, 30, 40)
        #print(outputs)     # shape=(1, 4, 512, 30, 40)
        #print(states)      # shape=(1, 512, 30, 40)
        if last_output.get_shape().ndims == 3:            #Nueva version
            last_output = K.expand_dims(last_output, dim=0)
        #if last_output.ndim == 3:
        #    last_output = K.expand_dims(last_output, dim=0)

        print("Red attentive convLSTM cargada")
        return last_output