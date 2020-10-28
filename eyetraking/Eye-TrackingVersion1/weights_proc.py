import tensorflow.keras.backend as K
from keras.engine import saving
import numpy as np

def load_weights_from_hdf5_group_new(f, layers, reshape=False):
    """Implements topological (order-based) weight loading.

    # Arguments
        f: A pointer to a HDF5 group.
        layers: a list of target layers.
        reshape: Reshape weights to fit the layer when the correct number
            of values are present but the shape does not match.

    # Raises
        ValueError: in case of mismatch between provided layers
            and weights file.
    """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None
    #Problemas en la m.layers[180].weights = [] (attentive)
    filtered_layers = []  #Recibe solo las layers que tienen pesos
    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = saving.load_attributes_from_hdf5_group(f, 'layer_names')
    filtered_layer_names = []
    for name in layer_names:
        g = f[name]
        weight_names = saving.load_attributes_from_hdf5_group(g, 'weight_names')
        if weight_names:
            filtered_layer_names.append(name)
    layer_names = filtered_layer_names
    #print("||||||||||||||||||||||||||||||||||||||")
    #for i in range(100,113):
    #    print(layer_names[i], "===", filtered_layers[i])
    #    print(" ")
    #print(layer_names[107], "===", filtered_layers[107]) #Problema en la 5/22
    #print("------------------------------------")
    #layer = filtered_layers[107]
    #symbolic_weights = layer.weights
    #for nro in range(0,22):
    #    print(nro)
    #    print(symbolic_weights[nro].shape)
    #print("||||||||||||||||||||||||||||||||||||||")
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight file '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' +
                         str(len(filtered_layers)) + ' layers.')

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = saving.load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights

        weight_values = saving.preprocess_weights_for_loading(layer,
                                                       weight_values,
                                                       original_keras_version,
                                                       original_backend,
                                                       reshape=reshape)
        #if(k == 107):
        #    for nro in range(0,21):
        #        print(weight_values[nro].shape)   
            
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) +
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
         
        #Zona Hack
        for i in range(len(symbolic_weights)):
            if (symbolic_weights[i].shape != weight_values[i].shape):
                weight_values[i] = np.moveaxis(weight_values[i], (0,1,2,3), (3,2,1,0))
        
        weight_value_tuples += zip(symbolic_weights, weight_values)

    #for i in range(len(weight_value_tuples)):
    #    print(i, weight_value_tuples[i][0].shape,"  =  ",weight_value_tuples[i][1].shape)

    K.batch_set_value(weight_value_tuples)
    print("Procedimiento weights_proc.py finalizado")

# with h5py.File('sam-resnet_salicon2017_weights.pkl', mode='r') as f:
#     if 'layer_names' not in f.attrs and 'model_weights' in f:
#         f = f['model_weights']
#     load_weights_from_hdf5_group_new(f, m.layers, reshape=True)
#     if hasattr(f, 'close'):
#         f.close()
#     elif hasattr(f.file, 'close'):
#         f.file.close()

