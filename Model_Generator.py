from keras.optimizers import SGD, Adam
from keras.applications import Xception, ResNet50
#from keras.applications import ResNet50
from keras.layers import Input, Dense, Flatten, Lambda, Multiply, Average, Add, Concatenate, Dot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from keras import backend as K

from scipy import interp
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from math import ceil
from itertools import cycle, product
from functools import partial

## Time
import datetime as dt
import time

import os 

from csv import writer

## Keras Steroids
from clr import LRFinder, OneCycleLR
import keras_metrics as km

import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ROC curve 
#Confusion

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

class Siamese():
    @staticmethod
    def initialize_weights(shape, name=None):
        """
            The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
            suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
        """
        return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
    @staticmethod
    def initialize_bias(shape, name=None):
        """
            The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
            suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
        """
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

    @staticmethod
    def weighted_categorical_crossentropy(y_true, y_pred, weights):
        # 0
        # 1 
        # 2 
        # weights[i, j] defines the weight for an example of class i 
        # which was falsely classified as class j.
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_pred, y_true) * final_mask

    batch_size = 16
    input_shape = (224, 224, 3)
    weights = None #rebuild on change
    conv_base = ResNet50(include_top=False, weights=weights, input_tensor=None, input_shape=input_shape, pooling=None, classes=None) #rebuild on change)
    conv_base_name = "resnet"
    freeze_conv_base = False
    output = "classification" #each change here should flip the reset_gen_flags
    siamese = True #each change here should flip the reset_gen_flags
    siamese_collector = "add"
    optimizer = SGD(lr=0.00001)
    epochs = 50
    model_object = None

    now_seconds = str(int(time.mktime(dt.datetime.now().timetuple())))

    path_to_csv = "training/training_final_siamese.csv"
    path_to_csv_test = "testing/testing_final_siamese.csv"
    data_dir = "training"
    test_dir = "testing"

    train_df = None
    val_df = None
    test_df = None
    
    reset_train_gen_flag = False
    reset_val_gen_flag = False
    reset_test_gen_flag = False

    model_list_history = []

    @classmethod
    def load_data(cls, use_val = True,  use_test_from_train = False,train_percentage = 0.6, 
        val_percentage = 0.3, test_percentage = 0.1 ):

        init_df = pd.read_csv(cls.path_to_csv, index_col="id")
        init_df = init_df.astype({'similarity':'str'})
        init_df = init_df.drop([1058, 2775])

        if use_test_from_train == False:
            cls.test_df = pd.read_csv(cls.path_to_csv_test, index_col="id")
            cls.test_df = cls.test_df.astype({'similarity':'str'})

        if use_val == True and use_test_from_train == True:
            train_cutoff_index = int(len(init_df) * train_percentage)
            cls.train_df = init_df[:train_cutoff_index]
            cls.test_df = init_df[train_cutoff_index:]
            val_cutoff_index = int(len(cls.test_df) * val_percentage)
            cls.val_df = cls.test_df[val_cutoff_index:]
            cls.test_df = cls.test_df[:val_cutoff_index]
            #cls.val_df.dis_label_class = labels_to_numbers(cls.val_df.dis_label_class)
            #val_label_class = to_categorical(cls.val_df.dis_label_class.values, num_classes=len(cls.val_df.dis_label_class.unique()))
            #val_similarity = to_categorical(cls.val_df.similarity.values, num_classes=len(cls.val_df.similarity.unique()))


        elif use_val == True and use_test_from_train == False:
            #val_cutoff_index = int(len(init_df) * val_percentage)
            cls.train_df = init_df[2716:]
            cls.val_df = init_df[:2716]

            ## test

            
        elif use_val == False and use_test_from_train == True:
            val_cutoff_index = int(len(init_df) * train_percentage)
            cls.train_df = init_df[:val_cutoff_index]
            cls.val_df = init_df[val_cutoff_index:]

            ## test
        elif use_val == False and use_test_from_train == False:
            ## No val or test from train
            cls.train_df = init_df

        print(len(cls.train_df.dis_label_class.unique()))

        cls.train_df.cpp_norm_reg = (cls.train_df.cpp_norm_reg - cls.train_df.cpp_norm_reg.mean())/cls.train_df.cpp_norm_reg.std()
        cls.train_df.poverty_reg = (cls.train_df.poverty_reg - cls.train_df.poverty_reg.mean())/cls.train_df.poverty_reg.std()
        if use_val:
            cls.val_df.cpp_norm_reg = (cls.val_df.cpp_norm_reg - cls.train_df.cpp_norm_reg.mean())/cls.train_df.cpp_norm_reg.std()
            cls.val_df.poverty_reg = (cls.val_df.poverty_reg - cls.train_df.poverty_reg.mean())/cls.train_df.poverty_reg.std()
        cls.test_df.cpp_norm_reg = (cls.test_df.cpp_norm_reg - cls.train_df.cpp_norm_reg.mean())/cls.train_df.cpp_norm_reg.std()
        cls.test_df.poverty_reg = (cls.test_df.poverty_reg - cls.train_df.poverty_reg.mean())/cls.train_df.poverty_reg.std()

        ## upsampling training set
        
        print("Training Set Raw (Disease)")
        print(cls.train_df.dis_label_class.value_counts())
        
        print("Training Set Raw (similarity)")
        print(cls.train_df.similarity.value_counts())



        df_train_low = cls.train_df[cls.train_df.dis_label_class == "low"]
        df_train_medium = cls.train_df[cls.train_df.dis_label_class == "medium"]
        df_train_high = cls.train_df[cls.train_df.dis_label_class == "high"]

        # Upsample minority class # Downsampling now 
        df_train_minority_medium_downsampled = resample(df_train_medium, 
                                    replace=True,     # sample with replacement
                                    n_samples=len(df_train_high ),    # to match majority class
                                    random_state=500) # reproducible results

        df_train_minority_low_downsampled = resample(df_train_low, 
                                    replace=True,     # sample with replacement
                                    n_samples=len(df_train_high ),    # to match majority class
                                    random_state=500) # reproducible results

        # Combine majority class with upsampled minority class
        cls.train_df = pd.concat([df_train_minority_low_downsampled, df_train_minority_medium_downsampled, df_train_high])
        
        print("Training set downsampled(Disease):")
        print(cls.train_df.dis_label_class.value_counts())

        print("Training set downsampled(Similarity):")
        print(cls.train_df.similarity.value_counts())

              
        print("Training set size: " + str(len(cls.train_df)))
        if use_val:
            print("Validation set size: " + str(len(cls.val_df)))
        print("Test set size: " + str(len(cls.test_df)))

    @classmethod
    def train_gen(cls, batch_size=batch_size, regression_columns = ["cpp_norm_reg", "poverty_reg"]):
        cls.reset_train_gen_flag = False #If we're here then reset is successful

        if cls.output == "classification" or cls.output == "mixed":
            train_classification_datagen = ImageDataGenerator(rescale=1./255)
        if cls.output == "regression" or cls.output == "mixed":
            train_regression_datagen = ImageDataGenerator(rescale=1./255)
        if cls.siamese:
            train_previous_datagen = ImageDataGenerator(rescale=1./255)
    
        if cls.output == "classification" or cls.output == "mixed":
        
            train_classification_generator = train_classification_datagen.flow_from_dataframe(cls.train_df, directory=cls.data_dir, x_col='filename_housing', y_col='dis_label_class', 
                            target_size=(224, 224), color_mode='rgb', classes=None, 
                            class_mode='categorical', batch_size=len(cls.train_df), shuffle=False, seed=500, 
                            save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                            interpolation='nearest', drop_duplicates=True)
        
        if cls.output == "regression" or cls.output == "mixed":

            train_regression_generator = train_regression_datagen.flow_from_dataframe(cls.train_df, directory=cls.data_dir, x_col='filename_housing', y_col=regression_columns, 
                    target_size=(224, 224), color_mode='rgb', classes=None, 
                    class_mode='other', batch_size=len(cls.train_df), shuffle=False, seed=500, 
                    save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                    interpolation='nearest', drop_duplicates=True)

        if cls.siamese:
            train_previous_generator = train_previous_datagen.flow_from_dataframe(cls.train_df, directory=cls.data_dir, x_col='filename_housing_previous', y_col='similarity', 
                            target_size=(224, 224), color_mode='rgb', classes=None, 
                            class_mode='binary', batch_size=len(cls.train_df), shuffle=False, seed=500, #batch_size = batch_size
                            save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                            interpolation='nearest', drop_duplicates=True)
        while True:
            if cls.reset_train_gen_flag:
                break
            if cls.output == "classification" or cls.output == "mixed":
                x_classification_train = train_classification_generator.next()
            if cls.output == "regression" or cls.output == "mixed":
                x_regression_train = train_regression_generator.next()
            if cls.siamese:
                x_previous_train = train_previous_generator.next()
            
            if cls.output == "classification" and cls.siamese:
                yield [x_previous_train[0], x_classification_train[0] ], [ x_classification_train[1],x_previous_train[1]]

            if cls.output == "regression" and cls.siamese:
                yield [x_previous_train[0], x_regression_train[0] ], [ x_regression_train[1],x_previous_train[1]]

            if cls.output == "mixed" and cls.siamese:
                yield [x_previous_train[0], x_classification_train[0] ], [ x_regression_train[1], x_classification_train[1],x_previous_train[1]]

            if cls.output == "classification" and not cls.siamese:
                yield [x_classification_train[0] ], [x_classification_train[1]]

            if cls.output == "regression" and not cls.siamese:
                yield [ x_regression_train[0] ], [ x_regression_train[1]]

            if cls.output == "mixed" and not cls.siamese:
                yield [x_classification_train[0] ], [ x_regression_train[1], x_classification_train[1]]

    @classmethod
    def val_gen(cls, batch_size=batch_size, regression_columns = ["cpp_norm_reg", "poverty_reg"]):
        cls.reset_val_gen_flag = False #If we're here then reset is successful

        if cls.output == "classification" or cls.output == "mixed":
            val_classification_datagen = ImageDataGenerator(rescale=1./255)
        if cls.output == "regression" or cls.output == "mixed":
            val_regression_datagen = ImageDataGenerator(rescale=1./255)
        if cls.siamese:
            val_previous_datagen = ImageDataGenerator(rescale=1./255)
    
        if cls.output == "classification" or cls.output == "mixed":
        
            val_classification_generator = val_classification_datagen.flow_from_dataframe(cls.val_df, directory=cls.data_dir, x_col='filename_housing', y_col='dis_label_class', 
                            target_size=(224, 224), color_mode='rgb', classes=None, 
                            class_mode='categorical', batch_size=len(cls.val_df), shuffle=False, seed=500, 
                            save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                            interpolation='nearest', drop_duplicates=True)
        
        if cls.output == "regression" or cls.output == "mixed":

            val_regression_generator = val_regression_datagen.flow_from_dataframe(cls.val_df, directory=cls.data_dir, x_col='filename_housing', y_col=regression_columns, 
                    target_size=(224, 224), color_mode='rgb', classes=None, 
                    class_mode='other', batch_size=len(cls.val_df), shuffle=False, seed=500, 
                    save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                    interpolation='nearest', drop_duplicates=True)

        if cls.siamese:
            val_previous_generator = val_previous_datagen.flow_from_dataframe(cls.val_df, directory=cls.data_dir, x_col='filename_housing_previous', y_col='similarity', 
                            target_size=(224, 224), color_mode='rgb', classes=None, 
                            class_mode='binary', batch_size=len(cls.val_df), shuffle=False, seed=500, 
                            save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                            interpolation='nearest', drop_duplicates=True)
        while True:
            if cls.reset_val_gen_flag:
                break
            if cls.output == "classification" or cls.output == "mixed":
                x_classification_val = val_classification_generator.next()
            if cls.output == "regression" or cls.output == "mixed":
                x_regression_val = val_regression_generator.next()
            if cls.siamese:
                x_previous_val = val_previous_generator.next()
            
            if cls.output == "classification" and cls.siamese:
                yield [x_previous_val[0], x_classification_val[0] ], [ x_classification_val[1], x_previous_val[1]]

            if cls.output == "regression" and cls.siamese:
                yield [x_previous_val[0], x_regression_val[0] ], [ x_regression_val[1], x_previous_val[1]]

            if cls.output == "mixed" and cls.siamese:
                yield [x_previous_val[0], x_classification_val[0] ], [ x_regression_val[1], x_classification_val[1],x_previous_val[1]]

            if cls.output == "classification" and not cls.siamese:
                yield [x_classification_val[0] ], [x_classification_val[1]]

            if cls.output == "regression" and not cls.siamese:
                yield [ x_regression_val[0] ], [ x_regression_val[1]]

            if cls.output == "mixed" and not cls.siamese:
                yield [x_classification_val[0] ], [ x_regression_val[1], x_classification_val[1]]
    
    @classmethod
    def test_gen(cls, batch_size=batch_size, regression_columns = ["cpp_norm_reg", "poverty_reg"]):
        cls.reset_test_gen_flag = False #If we're here then reset is successful

        if cls.output == "classification" or cls.output == "mixed":
            test_classification_datagen = ImageDataGenerator(rescale=1./255)
        if cls.output == "regression" or cls.output == "mixed":
            test_regression_datagen = ImageDataGenerator(rescale=1./255)
        if cls.siamese:
            test_previous_datagen = ImageDataGenerator(rescale=1./255)
    
        if cls.output == "classification" or cls.output == "mixed":
        
            test_classification_generator = test_classification_datagen.flow_from_dataframe(cls.test_df, directory=cls.test_dir, x_col='filename_housing', y_col='dis_label_class', 
                            target_size=(224, 224), color_mode='rgb', classes=None, 
                            class_mode='categorical', batch_size=len(cls.test_df), shuffle=False, seed=500, 
                            save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                            interpolation='nearest', drop_duplicates=True)
        
        if cls.output == "regression" or cls.output == "mixed":

            test_regression_generator = test_regression_datagen.flow_from_dataframe(cls.test_df, directory=cls.test_dir, x_col='filename_housing', y_col=regression_columns, 
                    target_size=(224, 224), color_mode='rgb', classes=None, 
                    class_mode='other', batch_size=len(cls.test_df), shuffle=False, seed=500, 
                    save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                    interpolation='nearest', drop_duplicates=True)

        if cls.siamese:
            test_previous_generator = test_previous_datagen.flow_from_dataframe(cls.test_df, directory=cls.test_dir, x_col='filename_housing_previous', y_col='similarity', 
                            target_size=(224, 224), color_mode='rgb', classes=None, 
                            class_mode='binary', batch_size=len(cls.test_df), shuffle=False, seed=500, 
                            save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                            interpolation='nearest', drop_duplicates=True)
        while True:
            if cls.reset_test_gen_flag:
                break
            if cls.output == "classification" or cls.output == "mixed":
                x_classification_test = test_classification_generator.next()
            if cls.output == "regression" or cls.output == "mixed":
                x_regression_test = test_regression_generator.next()
            if cls.siamese:
                x_previous_test = test_previous_generator.next()
            
            if cls.output == "classification" and cls.siamese:
                yield [x_previous_test[0], x_classification_test[0] ], [ x_classification_test[1], x_previous_test[1]]

            if cls.output == "regression" and cls.siamese:
                yield [x_previous_test[0], x_regression_test[0] ], [ x_regression_test[1], x_previous_test[1]]

            if cls.output == "mixed" and cls.siamese:
                yield [x_previous_test[0], x_classification_test[0] ], [ x_regression_test[1], x_classification_test[1],x_previous_test[1]]

            if cls.output == "classification" and not cls.siamese:
                yield [x_classification_test[0] ], [x_classification_test[1]]

            if cls.output == "regression" and not cls.siamese:
                yield [ x_regression_test[0] ], [ x_regression_test[1]]

            if cls.output == "mixed" and not cls.siamese:
                yield [x_classification_test[0] ], [ x_regression_test[1], x_classification_test[1]]

    @classmethod
    def build_model(cls):
        input_shape = cls.input_shape
        conv_base = cls.conv_base
        initialize_bias = cls.initialize_bias
        optimizer = cls.optimizer


        if cls.siamese == True:
    
            input_current = Input(shape= input_shape, name="input_current")
            input_prev = Input(shape= input_shape, name="input_prev")
            
            xception_model_embeddings_current = conv_base (input_current)
            xception_model_embeddings_previous = conv_base (input_prev)

            ## Add necessary max pooling and convs mentioned in the original thesis

            pre_L1_flatten = Flatten( name="flatten")(xception_model_embeddings_current)
            pre_L1_flatten_previous = Flatten(name="previous_flatten")(xception_model_embeddings_previous)

            L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]), name="l1")
            L1_distance = L1_layer([pre_L1_flatten_previous, pre_L1_flatten])
            similarity_prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias, name="similarity")(L1_distance)

            if cls.siamese_collector == "multiply":
                diffrentiable_conditional = Lambda( lambda x:K.switch(x>=0.5,Multiply()([xception_model_embeddings_current, xception_model_embeddings_previous]) , xception_model_embeddings_current))(similarity_prediction)
                
            elif cls.siamese_collector == "average":
                diffrentiable_conditional = Lambda( lambda x:K.switch(x>=0.5,Average()([xception_model_embeddings_current, xception_model_embeddings_previous]) , xception_model_embeddings_current))(similarity_prediction)
            
            elif cls.siamese_collector == "add":
                diffrentiable_conditional = Lambda( lambda x:K.switch(K.greater_equal(x,0.5),Add(name="add_inner")([xception_model_embeddings_current, xception_model_embeddings_previous]) ,xception_model_embeddings_current ), name="add_conditional")(similarity_prediction)

            #elif cls.siamese_collector == "concat":
            #    diffrentiable_conidtional_concatenate = Lambda( lambda x:K.switch(x>=0.5,Concatenate()([xception_model_embeddings_current, xception_model_embeddings_previous])  , xception_model_embeddings_current))(similarity_prediction)


            #elif cls.siamese_collector == "dot":
            #    diffrentiable_conditional_dot = Lambda( lambda x:K.switch(x>=0.5, Dot()([xception_model_embeddings_current, xception_model_embeddings_previous]), xception_model_embeddings_current))(similarity_prediction)


            #Here we try convolution

            #Be sure to initialize a different one with var input sizes for concat version
            #predictor_convolution = xception_conv_final_predictor(diffrentiable_conditional_add)
            dense_1 = Flatten() (diffrentiable_conditional)
            dense_2 = Dense(1024, activation='relu') (dense_1)
            dense_3 = Dense(512, activation='relu') (dense_2)
            dense_4 = Dense(256, activation='relu') (dense_3)

            if cls.output == "classification":
                final_prediction_class = Dense(3, activation='softmax', name="final_pred_class") (dense_4)
                ##input + output dict
                model_full = Model(inputs=[input_current, input_prev], outputs=[ final_prediction_class, similarity_prediction])
                #yield [x_previous_test[0], x_classification_test[0] ], [ x_classification_test[1], x_previous_test[1]]

                w_dis_label_class = np.ones((3, 3))
                w_dis_label_class[0, 1] = 2.
                w_dis_label_class[1, 0] = 2.
                w_dis_label_class[1, 2] = 2.
                w_dis_label_class[2, 1] = 2.
                w_dis_label_class_loss = partial(cls.weighted_categorical_crossentropy, weights=w_dis_label_class)
                w_dis_label_class_loss.__name__ = 'w_dis_label_class_loss'
                
                losses = {
                    'similarity':'binary_crossentropy',
                    'final_pred_class':w_dis_label_class_loss
                }
                
                loss_weights = { 
                  'similarity':5.,
                  'final_pred_class':1.
                }
                metrics = {
                    'similarity':['accuracy', km.binary_recall(), km.binary_precision()],
                    'final_pred_class':['accuracy', km.categorical_precision(), km.categorical_recall(), km.categorical_f1_score()]
                }
                
            elif cls.output == "regression":
                final_prediction_reg = Dense(2, name="final_pred_reg") (dense_4)
                model_full = Model(inputs=[input_current, input_prev], outputs=[ final_prediction_reg, similarity_prediction])
                losses = {
                    'similarity':'binary_crossentropy',
                    'final_pred_reg':'mae',
                }
                loss_weights = None
                metrics = {
                    'similarity':['accuracy', km.binary_recall(), km.binary_precision(), km.binary_f1_score()],
                    'final_pred_reg':['mae', 'mse', r2_keras],
                }
            elif cls.output == "mixed":
                final_prediction_reg = Dense(2, name="final_pred_reg") (dense_4)
                final_prediction_class = Dense(3, activation='softmax', name="final_pred_class") (dense_4)
                model_full = Model(inputs=[input_current, input_prev], outputs=[ final_prediction_reg, final_prediction_class, similarity_prediction])
                
                w_dis_label_class = np.ones((3, 3))
                w_dis_label_class[0, 1] = 2.
                w_dis_label_class[1, 0] = 2.
                w_dis_label_class[1, 2] = 2.
                w_dis_label_class[2, 1] = 2.
                w_dis_label_class_loss = partial(cls.weighted_categorical_crossentropy, weights=w_dis_label_class)
                w_dis_label_class_loss.__name__ = 'w_dis_label_class_loss'

                
                losses = {
                    'similarity':'binary_crossentropy',
                    'final_pred_reg':'mae',
                    'final_pred_class':w_dis_label_class_loss
                }
                
                loss_weights = { 
                  'similarity':5,
                  'final_pred_reg':1,
                  'final_pred_class':1
                }

                metrics = {
                    'similarity':['accuracy', km.binary_recall(), km.binary_precision(), km.binary_f1_score()],
                    'final_pred_reg':['mae', 'mse', r2_keras],
                    'final_pred_class':['accuracy', km.categorical_precision(), km.categorical_recall(), km.categorical_f1_score()]
                }
        else:
            input_current = Input(shape= cls.input_shape, name="input_current")
            xception_model_embeddings = cls.conv_base (input_current)
            flattened = Flatten( name="flatten")(xception_model_embeddings)
            dense_1 = Dense(1024, activation='relu') (flattened)
            dense_2 = Dense(512, activation='relu') (dense_1)
            dense_3 = Dense(256, activation='relu') (dense_2)

            if cls.output == "classification":
                final_prediction_class = Dense(3, activation='softmax', name="final_pred_class") (dense_3)
                model_full = Model([input_current], [final_prediction_class ])

                w_dis_label_class = np.ones((3, 3))
                w_dis_label_class[0, 1] = 2.
                w_dis_label_class[1, 0] = 2.
                w_dis_label_class[1, 2] = 2.
                w_dis_label_class[2, 1] = 2.

                w_dis_label_class_loss = partial(cls.weighted_categorical_crossentropy, weights=w_dis_label_class)
                w_dis_label_class_loss.__name__ = 'w_dis_label_class_loss'
                
                losses = {
                    'final_pred_class':w_dis_label_class_loss
                }
                
                loss_weights = None
                
                metrics = {
                    'final_pred_class':['accuracy', km.categorical_precision(), km.categorical_recall(), km.categorical_f1_score()]
                }

            elif cls.output == "regression":
                final_prediction_reg = Dense(2, name="final_pred_reg") (dense_3)
                model_full = Model([input_current], [final_prediction_reg ])
                losses = {
                    'final_pred_reg':'mae',
                }
                
                loss_weights = None
                
                metrics = {
                    'final_pred_reg':['mae', 'mse', r2_keras],
                }

            elif cls.output == "mixed":
                final_prediction_reg = Dense(2, name="final_pred_reg") (dense_3)
                final_prediction_class = Dense(3, activation='softmax', name="final_pred_class") (dense_3)
                model_full = Model([input_current], [ final_prediction_reg , final_prediction_class])
                
                w_dis_label_class = np.ones((3, 3))

                w_dis_label_class[0, 1] = 2.
                w_dis_label_class[1, 0] = 2.
                w_dis_label_class[1, 2] = 2.
                w_dis_label_class[2, 1] = 2.

                w_dis_label_class_loss = partial(cls.weighted_categorical_crossentropy, weights=w_dis_label_class)
                w_dis_label_class_loss.__name__ = 'w_dis_label_class_loss'

                losses = {
                    'final_pred_reg':'mae',
                    'final_pred_class':w_dis_label_class_loss
                }
                
                loss_weights = None

                metrics = {
                    'final_pred_reg':['mae', 'mse', r2_keras],
                    'final_pred_class':['accuracy', km.categorical_precision(), km.categorical_recall(), km.categorical_f1_score()]
                }

        cls.model_object =  model_full.compile(optimizer=optimizer,loss=losses, metrics=metrics, loss_weights=loss_weights)
        cls.model_object = model_full

    @classmethod
    def train_model(cls):
        if cls.freeze_conv_base == True:
            for layer in cls.conv_base.layers:
                layer.trainable = False
        else:
            for layer in cls.conv_base.layers:
                layer.trainable = True

        training_steps = ceil(len(cls.train_df)/cls.batch_size)
        validation_steps = ceil(len(cls.val_df)/cls.batch_size)
        epochs = cls.epochs

        if cls.siamese:
            siamese_status = "siamese"
        else:
            siamese_status = "baseline"
        folder_string = "{}_{}_{}".format(cls.conv_base_name, cls.output, siamese_status)
        folder_path = os.path.join(os.getcwd(), folder_string)

        if cls.siamese:
            pre_file_path = "{}/siamese-{}-".format(folder_path, cls.output)
        else:
            pre_file_path = "{}/baseline-{}-".format(folder_path, cls.output)

        post_file_path = "--resnet--{epoch:02d}-{val_loss:.2f}.hdf5"
        filepath= pre_file_path + post_file_path

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
        callbacks_list = [stopping, checkpoint]

        cls.model_list_history.append(cls.model_object.fit_generator(cls.train_gen(), steps_per_epoch=training_steps, epochs=epochs, verbose=1, callbacks=callbacks_list, validation_data=cls.val_gen(),
                                    validation_steps = validation_steps,class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, 
                                    shuffle=False))

    @classmethod
    def _plot_single_roc(cls, fpr_keras, tpr_keras, thresholds_keras, folder_path = "."):
        now = dt.datetime.now()
        now_seconds = str(int(time.mktime(now.timetuple())))
        auc_keras = auc(fpr_keras, tpr_keras)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        savestring1 = '{}/serial_{}_{}_roc.png'.format(folder_path, now_seconds, cls.output  )
        plt.savefig(savestring1)
        plt.show()

        plt.figure(2)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')
        save_string2 = '{}/serial_{}_{}_roc_zoom.png'.format(folder_path, now_seconds, cls.output  )
        plt.savefig(save_string2)
        plt.show()
        
    @classmethod
    def evaluate_model(cls):

        now_seconds = cls.now_seconds
        ## Putting reports into files so they don't crawl over each other
        if cls.siamese:
            siamese_status = "siamese"
        else:
            siamese_status = "baseline"
        folder_string = "{}_{}_{}".format(cls.conv_base_name, cls.output, siamese_status)
        folder_path = os.path.join(os.getcwd(), folder_string)

        output = cls.output

        if os.path.isdir(folder_path):
            pass
        else:
            os.mkdir(folder_path)

        if cls.output == "classification" and cls.siamese:
            #[x_previous_val[0], x_classification_val[0] ], [ x_classification_val[1], x_previous_val[1]]
            #gen_returned[x or y][input/output #][inputs/outputs]
            #model_full = Model(inputs=[input_current, input_prev], outputs=[ final_prediction_class, similarity_prediction])
            
            
            
            test_data = cls.test_gen().__next__()
            model_ys = cls.model_object.predict(test_data[0], batch_size=cls.batch_size, verbose=1, steps=None)
            class_preds = np.asarray(model_ys[0])
            sim_preds = np.asarray(model_ys[1])
            np.savetxt("{}/predicted_model_results_classes_{}_{}.csv".format(folder_path, output,"siamese" ), class_preds, delimiter=",")
            np.savetxt("{}/predicted_model_results_similarity_{}_{}.csv".format(folder_path, output,"siamese" ), sim_preds, delimiter=",")
            #print(classification_report(cls.test_df.dis_label_class, np.argmax(model_ys[0],axis=1), labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False))
            #yield [x_previous_test[0], x_classification_test[0] ], [ x_classification_test[1], x_previous_test[1]]
            #print(classification_report(cls.test_df.similarity, model_ys[1], labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False))
            pred_labels = (model_ys[1] < 0.5).astype(np.int)
            classification_report_similarity = classification_report(test_data[1][1], pred_labels)
            print(classification_report_similarity)
            classification_report_dis_labels = classification_report(np.argmax(test_data[1][0],axis=1), np.argmax(model_ys[0],axis=1))
            print(classification_report_dis_labels)
            confusion_matrix_similarity = confusion_matrix(test_data[1][1], pred_labels)
            print(confusion_matrix_similarity)
            cls._plot_confusion_matrix(confusion_matrix_similarity, ["0", "1"], folder_path)
            confusion_matrix_dis_labels = confusion_matrix(np.argmax(test_data[1][0],axis=1), np.argmax(model_ys[0],axis=1))
            print(confusion_matrix_dis_labels)
            cls._plot_confusion_matrix(confusion_matrix_dis_labels, ["low", "medium", "high"], folder_path)
            
            ### ROC binary 
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_data[1][1], model_ys[1].ravel())
            cls._plot_single_roc(fpr_keras, tpr_keras, thresholds_keras, folder_path)

            ### ROC multiclass
            binary_test_data = label_binarize(test_data[1][0], ["low", "medium", "high"])
            binary_pred_data = []
            for row_index in range(model_ys[0].shape[0]):
                if np.argmax(model_ys[0][row_index],axis=0) == 0:
                    binary_pred_data.append([1.,0.,0.])
                elif np.argmax(model_ys[0][row_index],axis=0) == 1:
                    binary_pred_data.append([0.,1.,0.])

                elif np.argmax(model_ys[0][row_index],axis=0) == 2:
                    binary_pred_data.append([0.,0.,1.])

            binary_pred_data = np.asarray(binary_pred_data)
            cls._plot_multi_label_roc(3, binary_test_data, binary_pred_data , 2, folder_path)

            gen_output = cls.test_gen().__next__()
            
            model_evaluation = cls.model_object.evaluate(x=gen_output[0], y=gen_output[1], batch_size=None, verbose=1, sample_weight=None, steps=None)
            print(model_evaluation)
            cls.export_csv(model_evaluation, folder_path)
    
            model_predicted_values = cls.model_object.predict(gen_output[0])
            model_predicted_values = pd.DataFrame(data=model_predicted_values,
                index=None,
                columns=None)

            if cls.siamese:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path,now_seconds, cls.output ,'siamese')
            else:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path, now_seconds, cls.output ,'baseline')

            model_predicted_values.to_csv(savestring)
        elif cls.output == "classification" and not cls.siamese:
            #[x_classification_test[0] ], [x_classification_test[1]]
            #gen_returned[x or y][input/output #][inputs/outputs]
            test_data = cls.test_gen().__next__()
            model_ys = cls.model_object.predict(test_data[0], batch_size=cls.batch_size, verbose=1, steps=None)
            np.savetxt("{}/predicted_model_results_{}_{}.csv".format(folder_path, output,"baseline" ), model_ys, delimiter=",")
            #print(classification_report(cls.test_df.dis_label_class, np.argmax(model_ys,axis=1).astype(str), labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False))
            #print(confusion_matrix(cls.test_df.dis_label_class, np.argmax(model_ys,axis=1).astype(str)))
            
            #pred_labels = (model_ys < 0.5).astype(np.int)
            classification_report_dis_labels = classification_report(test_data[1][0].argmax(axis=1), model_ys.argmax(axis=1))
            print(classification_report_dis_labels)
            confusion_matrix_dis_labels = confusion_matrix(test_data[1][0].argmax(axis=1), model_ys.argmax(axis=1))
            print(confusion_matrix_dis_labels)
            cls._plot_confusion_matrix(confusion_matrix_dis_labels, ["low", "medium", "high"], folder_path)

            binary_test_data = label_binarize(test_data[1][0], ["low", "medium", "high"])
            binary_pred_data = []
            for row_index in range(model_ys.shape[0]):
                if np.argmax(model_ys[row_index],axis=0) == 0:
                    binary_pred_data.append([1.,0.,0.])
                elif np.argmax(model_ys[row_index],axis=0) == 1:
                    binary_pred_data.append([0.,1.,0.])

                elif np.argmax(model_ys[row_index],axis=0) == 2:
                    binary_pred_data.append([0.,0.,1.])

            binary_pred_data = np.asarray(binary_pred_data)     
            cls._plot_multi_label_roc(3,binary_test_data,binary_pred_data, 2, folder_path  )       
            gen_output = cls.test_gen().__next__()
            model_evaluation = cls.model_object.evaluate(x=gen_output[0], y=gen_output[1], batch_size=None, verbose=1, sample_weight=None, steps=None)
            print(model_evaluation)
            cls.export_csv(model_evaluation, folder_path)

            model_predicted_values = cls.model_object.predict(gen_output[0])
            model_predicted_values = pd.DataFrame(data=model_predicted_values,
                index=None,
                columns=None)

            if cls.siamese:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path,now_seconds, cls.output ,'siamese')
            else:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path, now_seconds, cls.output ,'baseline')

            model_predicted_values.to_csv(savestring)

        ## Regression Losses

        if cls.output == "regression" and cls.siamese:
            #[x_previous_test[0], x_regression_test[0] ], [ x_regression_test[1], x_previous_test[1]]
            #gen_returned[x or y][input/output #][inputs/outputs]
            test_data = cls.test_gen().__next__()
            model_ys = cls.model_object.predict(test_data[0], batch_size=cls.batch_size, verbose=1, steps=None)
            #model_full = Model(inputs=[input_current, input_prev], outputs=[ final_prediction_reg, similarity_prediction])
            reg_preds = np.asarray(model_ys[0])
            sim_preds = np.asarray(model_ys[1])
            np.savetxt("{}/predicted_model_results_regression_{}_{}.csv".format(folder_path, output,"siamese" ), reg_preds, delimiter=",")
            np.savetxt("{}/predicted_model_results_similarity_{}_{}.csv".format(folder_path, output,"siamese" ), sim_preds, delimiter=",")

            pred_labels = (model_ys[1] < 0.5).astype(np.int)

            classification_report_similarity = classification_report(test_data[1][1], pred_labels)
            confusion_matrix_similarity = confusion_matrix(test_data[1][1], pred_labels)

            print(confusion_matrix_similarity)
            cls._plot_confusion_matrix(confusion_matrix_similarity, ["0", "1"], folder_path)

            #print(confusion_matrix(cls.test_df.similarity, model_ys[1].astype(str)))
            fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_data[1][1], model_ys[1].ravel())
            cls._plot_single_roc(fpr_keras, tpr_keras, thresholds_keras, folder_path)
            gen_output = cls.test_gen().__next__()
            #####
            model_evaluation = cls.model_object.evaluate(x=gen_output[0], y=gen_output[1], batch_size=None, verbose=1, sample_weight=None, steps=None)
            print(model_evaluation)
            cls.export_csv(model_evaluation, folder_path)

            model_predicted_values = cls.model_object.predict(gen_output[0])
            model_predicted_values = pd.DataFrame(data=model_predicted_values,
                index=None,
                columns=None)

            if cls.siamese:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path,now_seconds, cls.output ,'siamese')
            else:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path, now_seconds, cls.output ,'baseline')

            model_predicted_values.to_csv(savestring)

        elif cls.output == "regression" and not cls.siamese:
            #[ x_regression_test[0] ], [ x_regression_test[1]]
            #gen_returned[x or y][input/output #][inputs/outputs]
            gen_output = cls.test_gen().__next__()
            model_evaluation = cls.model_object.evaluate(x=gen_output[0], y=gen_output[1], batch_size=None, verbose=1, sample_weight=None, steps=None)
            print(model_evaluation)
            cls.export_csv(model_evaluation, folder_path)
            model_predicted_values = cls.model_object.predict(gen_output[0])
            np.savetxt("{}/predicted_model_results_{}_{}.csv".format(folder_path, output,"baseline" ), model_predicted_values, delimiter=",")

            model_predicted_values = pd.DataFrame(data=model_predicted_values,
                index=None,
                columns=None)

            if cls.siamese:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path,now_seconds, cls.output ,'siamese')
            else:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path, now_seconds, cls.output ,'baseline')

            model_predicted_values.to_csv(savestring)

        ## Mixed Models

        if cls.output == "mixed" and cls.siamese:
            #[x_previous_test[0], x_classification_test[0] ], [ x_regression_test[1], x_classification_test[1],x_previous_test[1]]
            #gen_returned[x or y][input/output #][inputs/outputs]
            #Model(inputs=[input_current, input_prev], outputs=[ final_prediction_reg, final_prediction_class, similarity_prediction])
            gen_output = cls.test_gen().__next__()
            model_ys = cls.model_object.predict(gen_output[0], batch_size=cls.batch_size, verbose=1, steps=None)
            np.savetxt("{}/predicted_model_results_{}_{}.csv".format(folder_path, output,"siamese" ), model_ys, delimiter=",")

            classification_report_similarity = classification_report(np.argmax(test_data[1][2],axis=1), np.argmax(model_ys[2],axis=1))
            print(classification_report_similarity)
            pred_labels = (model_ys[1] < 0.5).astype(np.int)
            classification_report_dis_labels = classification_report(test_data[1][1], pred_labels)
            print(classification_report_dis_labels)
            confusion_matrix_similarity = confusion_matrix(np.argmax(test_data[1][2],axis=1), np.argmax(model_ys[2],axis=1))
            print(confusion_matrix_similarity)
            cls._plot_confusion_matrix(confusion_matrix_similarity, ["0", "1"], folder_path)
            confusion_matrix_dis_labels = confusion_matrix(test_data[1][1], pred_labels)
            print(confusion_matrix_dis_labels)
            cls._plot_confusion_matrix(confusion_matrix_dis_labels, ["low", "medium", "high"], folder_path)

            fpr_keras, tpr_keras, thresholds_keras = roc_curve(cls.test_df.similarity, model_ys[2].astype(str).ravel())
            cls._plot_single_roc(fpr_keras, tpr_keras, thresholds_keras, folder_path)

            binary_test_data = label_binarize(test_data[1][0], ["low", "medium", "high"])
            binary_pred_data = []
            for row_index in range(model_ys[0].shape[0]):
                if np.argmax(model_ys[0][row_index],axis=0) == 0:
                    binary_pred_data.append([1.,0.,0.])
                elif np.argmax(model_ys[0][row_index],axis=0) == 1:
                    binary_pred_data.append([0.,1.,0.])

                elif np.argmax(model_ys[0][row_index],axis=0) == 2:
                    binary_pred_data.append([0.,0.,1.])

            binary_pred_data = np.asarray(binary_pred_data)

            cls._plot_multi_label_roc(3, binary_test_data, binary_pred_data, 2, folder_path)       
            model_evaluation = cls.model_object.evaluate(x=gen_output[0], y=gen_output[1], batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
            print(model_evaluation)
            cls.export_csv(model_evaluation, folder_path)
            model_predicted_values = cls.model_object.predict(gen_output[0])
            model_predicted_values = pd.DataFrame(data=model_predicted_values,
                index=None,
                columns=None)

            if cls.siamese:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path,now_seconds, cls.output ,'siamese')
            else:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path, now_seconds, cls.output ,'baseline')

            model_predicted_values.to_csv(savestring)
            del gen_output
        elif cls.output == "mixed" and not cls.siamese:
            #[x_classification_test[0] ], [ x_regression_test[1], x_classification_test[1]]
            #gen_returned[x or y][input/output #][inputs/outputs]
            #model_full = Model([input_current], [ final_prediction_reg , final_prediction_class])
                        
            test_data = cls.test_gen().__next__()
            model_ys = cls.model_object.predict(test_data[0], batch_size=cls.batch_size, verbose=1, steps=None)
            np.savetxt("{}/predicted_model_results_{}_{}.csv".format(folder_path, output,"baseline" ), model_ys, delimiter=",")

            pred_labels = (model_ys[1] < 0.5).astype(np.int)
            classification_report_dis_labels = classification_report(test_data[1][1], pred_labels)
            print(classification_report_dis_labels)

            cls._plot_confusion_matrix(confusion_matrix_similarity, ["0", "1"], folder_path)
            confusion_matrix_dis_labels = confusion_matrix(test_data[1][1], pred_labels)
            print(confusion_matrix_dis_labels)

            binary_test_data = label_binarize(test_data[1][1], ["low", "medium", "high"])
            binary_pred_data = []
            for row_index in range(model_ys[1].shape[0]):
                if np.argmax(model_ys[1][row_index],axis=0) == 0:
                    binary_pred_data.append([1.,0.,0.])
                elif np.argmax(model_ys[1][row_index],axis=0) == 1:
                    binary_pred_data.append([0.,1.,0.])

                elif np.argmax(model_ys[1][row_index],axis=0) == 2:
                    binary_pred_data.append([0.,0.,1.])

            binary_pred_data = np.asarray(binary_pred_data)
            cls._plot_multi_label_roc(3, binary_test_data, binary_pred_data, 2, folder_path)       
            model_evaluation = cls.model_object.evaluate(x=gen_output[0], y=gen_output[1], batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
            print(model_evaluation)
            cls.export_csv(model_evaluation, folder_path)
            model_predicted_values = cls.model_object.predict(gen_output[0])
            model_predicted_values = pd.DataFrame(data=model_predicted_values,
                index=None,
                columns=None)

            if cls.siamese:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path,now_seconds, cls.output ,'siamese')
            else:
                savestring = "{}/predicted_{}_{}_{}.csv".format(folder_path, now_seconds, cls.output ,'baseline')

            model_predicted_values.to_csv(savestring)
            
    @classmethod
    def plot_learning_rate(cls):
        counter = 1
        for iteration_number, iteration_history in enumerate(cls.model_list_history):
            iteration_number += 1
            prefix = "Training iteration {}".format(iteration_number)
            for metric_name, metric_values in iteration_history.history.items():
                plt.figure(counter)
                epochs = range(1, len(metric_values) + 1)
                plt.plot(epochs, metric_values)
                plt.title(prefix)
                plt.xlabel(prefix + ' epochs')
                plt.ylabel(metric_name)
                counter += 1
        plt.show()

    
    @classmethod
    def plot_keras_model(cls):
        if cls.siamese:
            desc_string = "siamese"
        else:
            desc_string = "baseline"
        plot_model(cls.model_object, '{}_{}.png'.format(cls.output, desc_string))

    @classmethod
    def _plot_confusion_matrix(cls, cm, classes, folder_path = "."):
        cm = cm / cm.astype(np.float).sum(axis=1)

        cmap=plt.cm.Blues
        now = dt.datetime.now()
        now_seconds = str(int(time.mktime(now.timetuple())))

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title="Confusion Matrix {}".format(cls.output),
           ylabel='True label',
           xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = 'f' #FIXME?
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        savestring = '{}/{}_{}_confusion_matrix.png'.format(folder_path, now_seconds, cls.output)
        plt.savefig(savestring)
        plt.show()
        

    
    @classmethod
    # Thanks to https://github.com/Tony607/ROC-Keras/blob/master/ROC-Keras.ipynb
    def _plot_multi_label_roc(cls,n_classes, ground_truth, keras_preds, lw=2, folder_path = "."):

        now = dt.datetime.now()
        now_seconds = str(int(time.mktime(now.timetuple())))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        #We need to output to categorical first
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(ground_truth[:, i], keras_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(ground_truth.ravel(), keras_preds.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(1)
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        savestring1 = "{}/{}_{}_multi_roc.png".format(folder_path, now_seconds, cls.output)
        plt.savefig(savestring1)
        plt.show()


        # Zoom in view of the upper left corner.
        plt.figure(2)
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
        plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        savestring2 = "{}/{}_{}_multi_roc_zoom.png".format(folder_path, now_seconds, cls.output)
        plt.savefig(savestring2)
        plt.show()

    @classmethod
    def find_lr_rate(cls, search_for = "lr"):
        number_of_samples = len(cls.train_df)
        batch_size = cls.batch_size

        training_steps = ceil(len(cls.train_df)/cls.batch_size)
        validation_steps = ceil(len(cls.val_df)/cls.batch_size)

        if search_for == "lr":
            lr_finder = LRFinder(number_of_samples, batch_size, minimum_lr=1e-5, maximum_lr=10.,
                        lr_scale='exp',
                        validation_data=None,  # use the validation data for losses
                        #validation_sample_rate=5,
                        save_dir='weights/', verbose=True)

            cls.model_object.fit_generator(cls.train_gen(), steps_per_epoch=training_steps, epochs=1, verbose=1, callbacks=[lr_finder], validation_data=cls.val_gen(),
                                        validation_steps = validation_steps,class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, 
                                        shuffle=False)
        elif search_for == "momentum":
            MOMENTUMS = [0.9, 0.95, 0.99]
            val_gen = cls.val_gen()
            cls.model_object.save_weights('pre_momentum.h5')
            for momentum in MOMENTUMS:
                cls.model_object.load_weights('pre_momentum.h5')

                lr_finder = LRFinder(number_of_samples, batch_size, minimum_lr=0.002, maximum_lr=0.02,
                            validation_data=None,
                            validation_sample_rate=5,
                            lr_scale='linear', save_dir='weights/momentum/momentum-%s/' % str(momentum),
                            verbose=True)
                
                cls.model_object.fit_generator(cls.train_gen(), steps_per_epoch=training_steps, epochs=1, verbose=1, callbacks=[lr_finder], validation_data=cls.val_gen(),
                                            validation_steps = validation_steps,class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, 
                                            shuffle=False)

        elif search_for == "weight_decay":

            WEIGHT_DECAY_FACTORS = [1e-7, 3e-7, 3e-6]
            val_gen = cls.val_gen()

            cls.model_object.save_weights('pre_momentum.h5')
            for weight_decay in WEIGHT_DECAY_FACTORS:

                cls.model_object.load_weights('pre_momentum.h5')                
                lr_finder = LRFinder(number_of_samples, batch_size, minimum_lr=0.002, maximum_lr=0.02,
                            validation_data=None,
                            validation_sample_rate=5,
                            lr_scale='linear', save_dir='weights/weight_decay/weight_decay-%s/' % str(weight_decay),
                            verbose=True)

                
    @classmethod
    def super_train_model(cls, max_lr_given = 0.02):

        num_samples = len(cls.train_df)
        batch_size = cls.batch_size
        num_epoch = cls.epochs
        max_lr =max_lr_given
        lr_manager = OneCycleLR( max_lr, 0.1, None, 0.95, 0.85)
        
        training_steps = ceil(len(cls.train_df)/cls.batch_size)
        validation_steps = ceil(len(cls.val_df)/cls.batch_size)

         ## Putting reports into files so they don't crawl over each other
        if cls.siamese:
            siamese_status = "siamese"
        else:
            siamese_status = "baseline"
        folder_string = "{}_{}_{}".format(cls.conv_base_name, cls.output, siamese_status)
        folder_path = os.path.join(os.getcwd(), folder_string)

        if os.path.isdir(folder_path):
            pass
        else:
            os.mkdir(folder_path)

        if cls.siamese:
            pre_file_path = "{}/siamese-{}-".format(folder_path, cls.output)
        else:
            pre_file_path = "{}/baseline-{}-".format(folder_path, cls.output)

        post_file_path = "--resnet--{epoch:02d}-{val_loss:.2f}.hdf5"
        filepath= pre_file_path + post_file_path

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
        callbacks_list = [lr_manager, checkpoint] #stopping, checkpoint]
        
        X_train, y_train = cls.train_gen().__next__()
        X_val, y_val = cls.val_gen().__next__()

        cls.model_list_history.append(cls.model_object.fit(x=X_train, y=y_train, batch_size=batch_size,
            epochs=num_epoch, verbose=1, callbacks=callbacks_list, validation_split=0.0, validation_data=(X_val, y_val), 
            shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, 
            steps_per_epoch=None, validation_steps=None))


    @classmethod
    def export_csv(cls, evaluation_result, folder_path = "."):
        now = dt.datetime.now()
        now_seconds = str(int(time.mktime(now.timetuple())))
        
        if cls.siamese:
            savestring1 = "{}/{}_{}_{}.csv".format(folder_path,now_seconds, cls.output ,'siamese')
        else:
            savestring1 = "{}/{}_{}_{}.csv".format(folder_path, now_seconds, cls.output ,'baseline')

        csvfd = open(savestring1, "w+")
        csvwriter = writer(csvfd, delimiter=',')

        csvwriter.writerow(cls.model_object.metrics_names)
        csvwriter.writerow(evaluation_result)

        csvfd.close()

