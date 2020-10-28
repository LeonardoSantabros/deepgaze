from __future__ import division

import os
os.chdir("C:\\Users\\alebj\\Documents\\Python Scripts\\CNN Eye-Tracking")

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras import Input     #Agregado

import cv2, sys
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
#from models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss
from models import sam_resnet, kl_divergence, correlation_coefficient, nss #New version
import h5py   #Agregado
from keras.engine import saving   #Agregado
import weights_proc               #Nuevo modulo creado

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

#Ejecucion por codigo Nueva version
listaArg = ["main.py", 'test', 'C:\\Users\\alebj\\Documents\\Python Scripts\\CNN Eye-Tracking\\sample_images']

##///////////////////////////////////////////////////////////
#Clase nueva agregada al codigo para agregar un nuevo metodo

class ModelAux(Model):     #NUEVA CLASE CREADA

    @saving.allow_read_from_gcs
    def load_weights_new(self, filepath,
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
            #Nueva funcion desarrollada
            weights_proc.load_weights_from_hdf5_group_new(f, self.layers, reshape=reshape)
            if hasattr(f, 'close'):
                f.close()
            elif hasattr(f.file, 'close'):
                f.file.close()
   
                
#class TensorNew(Layer):
#    '''We have the calls to add_weight(), and then call the super's build()'''
#    def __init__(self):
#        super().__init__()
#        self.tensor_shape = self.   #EN CONSTRUCCION
#
#    def numpy(self):
#        return tf.make_ndarray(self)

##//////////////////////////////////////////////////////////////////

def generator(b_s, phase_gen='train'):
    if phase_gen == 'train':
        images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_train_path + f for f in os.listdir(fixs_train_path) if f.endswith('.mat')]
    elif phase_gen == 'val':
        images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [fixs_val_path + f for f in os.listdir(fixs_val_path) if f.endswith('.mat')]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()
    fixs.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian], [Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + "\\" + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    #counter = 0
    #while True:
    #    yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
    #    counter = (counter + b_s) % len(images)

    #Funciona para b_s = 1  NUEVA VERSION!
    counter = 0        
    while counter < len(images):
        print("Ejecutado generator_test para la imagen ", counter + 1)
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = counter + 1
        
if __name__ == '__main__':
    #if len(sys.argv) == 1:                 #Ejecucion por consola de windows
    if len(listaArg) == 1:                  #Ejecucion por codigo
        raise NotImplementedError
    else:
        #phase = sys.argv[1]                #Ejecucion por consola de windows
        phase = listaArg[1]                 #Ejecucion por codigo
        #x = Input((3, shape_r, shape_c))
        x = Input((shape_r, shape_c, 3))    #Nueva version
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))
        #x_maps = Input((shape_r_gt, shape_c_gt, nb_gaussian))   #Nueva version
        
        if version == 0:   #NO USADO
        #    m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
            print("Not Compiling SAM-VGG")   #Nueva version
        #    m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
        elif version == 1:
            '''Hint of the problem: something is not the output of a keras layer. 
            You should put it in a lambda layer
            When invoking the Model API, the value for outputs argument should 
            be tensor(or list of tensors), in this case it is a list of list of 
            tensors, hence there is a problem'''
            #m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
            #m = Model(inputs=[x, x_maps], outputs=sam_resnet([x, x_maps]))  #New version
            m = ModelAux(inputs=[x, x_maps], outputs=sam_resnet([x, x_maps])) #Final version

            print("Compiling SAM-ResNet")
            m.compile(RMSprop(lr=1e-4), 
                      loss=[kl_divergence, correlation_coefficient, nss])
            print("Compilado")
        else:
            raise NotImplementedError

        if phase == 'train':
            if nb_imgs_train % b_s != 0 or nb_imgs_val % b_s != 0:
                print("The number of training and validation images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                exit()

            if version == 0:
                print("Training SAM-VGG")
                m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                                validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                                callbacks=[EarlyStopping(patience=3),
                                           ModelCheckpoint('weights.sam-vgg.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])
            elif version == 1:
                print("Training SAM-ResNet")
                m.fit_generator(generator(b_s=b_s), nb_imgs_train, nb_epoch=nb_epoch,
                                validation_data=generator(b_s=b_s, phase_gen='val'), nb_val_samples=nb_imgs_val,
                                callbacks=[EarlyStopping(patience=3),
                                           ModelCheckpoint('weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.pkl', save_best_only=True)])

        elif phase == "test":
            # Output Folder Path
            output_folder = 'predictions/'

            #if len(sys.argv) < 2:              #Ejecucion por consola de windows
            #    raise SyntaxError
            #imgs_test_path = sys.argv[2]
            
            if len(listaArg) < 2:               #Ejecucion por codigo
                raise SyntaxError
            imgs_test_path = listaArg[2]
            
            file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            file_names.sort()
            nb_imgs_test = len(file_names)

            if nb_imgs_test % b_s != 0:
                print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                exit()

            if version == 0:  #NO ACTIVA
                print("Not Loading SAM-VGG weights")
                #m.load_weights('weights/sam-vgg_salicon_weights.pkl')
            elif version == 1:
                
                # for i in range(len(m.layers)):
                #     print('____________________________________________')
                #     nro = i        
                #     print(i)
                #     print(m.layers[nro])
                #     weight = m.layers[nro].get_weights()
                #     if(len(weight) == 0):
                #         print("La layer no tiene pesos") 
                #     else:          
                #         weight0 = np.array(weight[1])
                #         print(weight0.shape)
                #     print('____________________________________________')
    
                print("Loading SAM-ResNet weights")
                #m.load_weights('weights/sam-resnet_salicon_weights.pkl')
                #m.load_weights('weights/sam-resnet_salicon2017_weights.pkl') #New version
                m.load_weights_new('weights/sam-resnet_salicon2017_weights.pkl', reshape=True) #Final version
                print("==============================================")
                
            #Todo controlado hasta aqui\\\\\\\\\\\\\\\\\\\\\\\\\\
            print("Predicting saliency maps for " + imgs_test_path)
            '''https://stackoverflow.com/questions/58352326/running-the-tensorflow-2-0-code-gives-valueerror-tf-function-decorated-functio'''
            #predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)[0]
            #predictions = m.predict(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)[0] #Nueva version      
            #predictions = m.predict(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test) #Nueva version
            predictions = m.predict(generator_test(b_s=b_s, imgs_test_path=imgs_test_path),steps = nb_imgs_test)[0] #Nueva version. Output shape = (1, 1, 480, 640)
            print("Longitud de `predictions`: ", len(predictions))
            
            #x = [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
            #predictions = m.predict(x,batch_size = nb_imgs_test)[0] #PRUEBAS

            print("==============================================")
            for pred, name in zip(predictions, file_names):
                #pred = predictions[0]
                #name = file_names[0] 
                print("Dibujando el saliency map de la imagen ", name)
                #original_image = cv2.imread(imgs_test_path + name, 0)
                original_image = cv2.imread(imgs_test_path + "/" + name, 0)
                #res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
                res = postprocess_predictions(pred, original_image.shape[0], original_image.shape[1]) #New version
                cv2.imwrite(output_folder + '%s' % name, res.astype(int)) #res.shape (300, 450)
        else:
            raise NotImplementedError
            
print("Programa finalizado")
