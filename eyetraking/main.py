from __future__ import division

from flask import Flask, render_template
import os, sys, json, shutil
# newpath = os.getcwd() + "/Eye-TrackingVersion1" # version linea de comandos
newpath = os.getcwd() + "/eyetraking/Eye-TrackingVersion1" # version api
os.chdir(newpath)
sys.path.append('/root/miniconda2/lib/python2.7/site-packages/')
# print os.path.abspath()
# print newpath
# print sys.path

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input
from keras.models import Model
import cv2, sys
import numpy as np
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss

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
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    print( json.dumps(images) )

    counter = 0
    while True:
        yield [preprocess_images(images[counter:counter + b_s], shape_r, shape_c), gaussian]
        counter = (counter + b_s) % len(images)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise NotImplementedError
    else:
        phase = sys.argv[1]
        x = Input((3, shape_r, shape_c))
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

        if version == 0:
            m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
            print("Compiling SAM-VGG")
            m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
        elif version == 1:
            m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
            print("Compiling SAM-ResNet")
            m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
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

            if len(sys.argv) < 2:
                raise SyntaxError
            imgs_test_path = sys.argv[2]

            file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            file_names.sort()
            nb_imgs_test = len(file_names)

            if nb_imgs_test % b_s != 0:
                print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
                exit()

            if version == 0:
                print("Loading SAM-VGG weights")
                m.load_weights('weights/sam-vgg_salicon_weights.pkl')
            elif version == 1:
                print("Loading SAM-ResNet weights")
                #m.load_weights('weights/sam-resnet_salicon_weights.pkl') 
                m.load_weights('weights/sam-resnet_salicon2017_weights.pkl') #Nueva version

            print("Predicting saliency maps for " + imgs_test_path)
            predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)[0]


            for pred, name in zip(predictions, file_names):
                original_image = cv2.imread(imgs_test_path + name, 0)
                res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
                print(pred, name)
                print( '/code/eyetraking/' + output_folder + '%s' % name, res.astype(int) )
                cv2.imwrite( '/code/eyetraking/' + output_folder + '%s' % name, res.astype(int))
        else:
            raise NotImplementedError

def main():
    file_names = [f for f in os.listdir('/code/eyetraking/sample_images') if f.endswith(('.jpg', '.jpeg', '.png'))]
    prediction_names = [f for f in os.listdir('/code/eyetraking/predictions') if f.endswith(('.jpg', '.jpeg', '.png'))]
    y = json.dumps(file_names)
    return render_template('home.html', images=file_names, predictions=prediction_names)

def predict():
    
    x = Input((3, shape_r, shape_c))
    x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

    if version == 0:
        m = Model(input=[x, x_maps], output=sam_vgg([x, x_maps]))
        print("Compiling SAM-VGG")
        m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
    elif version == 1:
        m = Model(input=[x, x_maps], output=sam_resnet([x, x_maps]))
        print("Compiling SAM-ResNet")
        m.compile(RMSprop(lr=1e-4), loss=[kl_divergence, correlation_coefficient, nss])
    else:
        raise NotImplementedError

    
    # Output Folder Path
    output_folder = 'predictions/'

    imgs_test_path = "/code/eyetraking/sample_images/"

    file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    file_names.sort()
    nb_imgs_test = len(file_names)

    if nb_imgs_test % b_s != 0:
        print("The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly.")
        exit()

    if version == 0:
        print("Loading SAM-VGG weights")
        m.load_weights('weights/sam-vgg_salicon_weights.pkl')
    elif version == 1:
        print("Loading SAM-ResNet weights")
        #m.load_weights('weights/sam-resnet_salicon_weights.pkl') 
        m.load_weights('weights/sam-resnet_salicon2017_weights.pkl') #Nueva version

    print("Predicting saliency maps for " + imgs_test_path)
    predictions = m.predict_generator(generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test)[0]


    for pred, name in zip(predictions, file_names):
        original_image = cv2.imread(imgs_test_path + name, 0)
        res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
        print(pred, name)
        print( '/code/eyetraking/' + output_folder + '%s' % name, res.astype(int) )
        cv2.imwrite( '/code/eyetraking/' + output_folder + '%s' % name, res.astype(int))
        
    images = []
    predicts = []
    diccionario = {}

    for path in file_names: 
        images.append( os.path.join("/eyetraking/sample_images/", path) )
        print images

    diccionario["imagenes"] = images
        
    for path in file_names: 
        predicts.append( os.path.join("/eyetraking/predictions/", path) )
        print predicts

    diccionario["predicts"] = predicts

    return diccionario

def uploadImg():
    imgs_path = "/code/eyetraking/sample_images/"
    predictions_path = "/code/eyetraking/predictions/"
    for files in os.listdir(imgs_path):
        path = os.path.join(imgs_path, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

    for files in os.listdir(predictions_path):
        path = os.path.join(predictions_path, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)