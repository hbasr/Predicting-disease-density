#!/usr/bin/env python
# coding: utf-8

# In the beginning there was the repo, the csv file floated over the data, we took it, made it into a train + test split and passed it on to the next segment

# In[5]:


get_ipython().system('git clone https://mekaneeky:splashscreen123!@github.com/mekaneeky/pestilence.git')


# In[6]:


from keras.utils import to_categorical

import pandas as pd
import numpy as np

def labels_to_numbers(df_column):
    numbers_dict = {value:number for (number, value) in enumerate(df_column.unique())}
    return df_column.apply( lambda x : numbers_dict[x])


path_to_csv = "pestilence/training/training_final.csv"
path_to_csv_test = "pestilence/testing/testing_final.csv"
init_df = pd.read_csv(path_to_csv, index_col="id")
use_val = True
use_test_from_train = False
if use_test_from_train == False:
    test_df = pd.read_csv(path_to_csv_test, index_col="id")
#balady
#index = init_df['similarity'].index[init_df['similarity'].apply(np.isnan)]
init_df = init_df.drop([1059, 2776])
#Data is sequential so no shuffling
train_percentage = 0.6
val_percentage = 0.3
test_percentage = 0.1
#train_df.dis_label_class = labels_to_numbers(train_df.dis_label_class)
#train_label_class = to_categorical(train_df.dis_label_class.values, num_classes=len(train_df.dis_label_class.unique()))
#train_similarity = to_categorical(train_df.similarity.values, num_classes=len(train_df.similarity.unique()))

if use_val == True and use_test_from_train == True:
    train_cutoff_index = int(len(init_df) * train_percentage)
    train_df = init_df[:train_cutoff_index]
    test_df = init_df[train_cutoff_index:]
    val_cutoff_index = int(len(test_df) * val_percentage)
    val_df = test_df[val_cutoff_index:]
    test_df = test_df[:val_cutoff_index]
    #val_df.dis_label_class = labels_to_numbers(val_df.dis_label_class)
    #val_label_class = to_categorical(val_df.dis_label_class.values, num_classes=len(val_df.dis_label_class.unique()))
    #val_similarity = to_categorical(val_df.similarity.values, num_classes=len(val_df.similarity.unique()))


elif use_val == True and use_test_from_train == False:
    val_cutoff_index = int(len(init_df) * val_percentage)
    train_df = init_df[:val_cutoff_index]
    val_df = init_df[val_cutoff_index:]

    ## test

    
elif use_val == False and use_test_from_train == True:
    val_cutoff_index = int(len(init_df) * train_percentage)
    train_df = init_df[:val_cutoff_index]
    val_df = init_df[val_cutoff_index:]

    ## test
elif use_val == False and use_test_from_train == False:
    ## No val or test from train
    train_df = init_df

print(len(train_df.dis_label_class.unique()))

train_df.cpp_norm_reg = (train_df.cpp_norm_reg - train_df.cpp_norm_reg.mean())/train_df.cpp_norm_reg.std()
train_df.poverty_reg = (train_df.poverty_reg - train_df.poverty_reg.mean())/train_df.poverty_reg.std()
if use_val:
    val_df.cpp_norm_reg = (val_df.cpp_norm_reg - train_df.cpp_norm_reg.mean())/train_df.cpp_norm_reg.std()
    val_df.poverty_reg = (val_df.poverty_reg - train_df.poverty_reg.mean())/train_df.poverty_reg.std()
test_df.cpp_norm_reg = (test_df.cpp_norm_reg - train_df.cpp_norm_reg.mean())/train_df.cpp_norm_reg.std()
test_df.poverty_reg = (test_df.poverty_reg - train_df.poverty_reg.mean())/train_df.poverty_reg.std()

print("Training set size: " + str(len(train_df)))
if use_val:
    print("Validation set size: " + str(len(val_df)))
print("Test set size: " + str(len(test_df)))





# Now we have our training, testing and possibly validation sets. Now we need to create a generator to pass the data to the model

# In[7]:


from keras.preprocessing.image import ImageDataGenerator

data_dir = "pestilence/training"
if use_test_from_train == False:
    test_dir = "pestilence/testing"
else:
    test_dir = data_dir
batch_size = 16


train_datagen = ImageDataGenerator(rescale=1./255)
train_previous_datagen = ImageDataGenerator(rescale=1./255)
if use_val:
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_previous_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
test_previous_datagen = ImageDataGenerator(rescale=1./255)


def train_gen( batch_size=batch_size, regression_columns = ["cpp_norm_reg", "poverty_reg"]):
    
    train_generator = train_datagen.flow_from_dataframe(train_df, directory=data_dir, x_col='filename_housing', y_col='dis_label_class', 
                    target_size=(224, 224), color_mode='rgb', classes=None, 
                    class_mode='categorical', batch_size=batch_size, shuffle=False, seed=500, 
                    save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                    interpolation='nearest', drop_duplicates=True)
    
    train_previous_generator = train_datagen.flow_from_dataframe(train_df, directory=data_dir, x_col='filename_housing', y_col=regression_columns, 
                    target_size=(224, 224), color_mode='rgb', classes=None, 
                    class_mode='other', batch_size=batch_size, shuffle=False, seed=500, 
                    save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                    interpolation='nearest', drop_duplicates=True)
    while True:
      x_train = train_generator.next()
      x_previous_train = train_previous_generator.next()
      yield [x_train[0] ], [x_previous_train[1], x_train[1]]
    
if use_val:
  def val_gen( batch_size= batch_size, regression_columns = ["cpp_norm_reg", "poverty_reg"]):

      val_generator = val_datagen.flow_from_dataframe(test_df, directory=test_dir, x_col='filename_housing', y_col='dis_label_class', 
                      target_size=(224, 224), color_mode='rgb', classes=None, 
                      class_mode='categorical', batch_size=batch_size, shuffle=False, seed=500, 
                      save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                      interpolation='nearest', drop_duplicates=True)

      val_previous_generator = val_previous_datagen.flow_from_dataframe(test_df, directory=test_dir, x_col='filename_housings', y_col=regression_columns, 
                      target_size=(224, 224), color_mode='rgb', classes=None, 
                      class_mode='other', batch_size=batch_size, shuffle=False, seed=500, 
                      save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                      interpolation='nearest', drop_duplicates=False)
      while True:
        x_val = val_generator.next()
        x_previous_val = val_previous_generator.next()
        yield [x_val[0] ], [x_previous_val[1], x_val[1]]
      
def test_gen( batch_size=batch_size, regression_columns = ["cpp_norm_reg", "poverty_reg"]):
    
    test_generator = test_datagen.flow_from_dataframe(test_df, directory=test_dir, x_col='filename_housing', y_col='dis_label_class', 
                    target_size=(224, 224), color_mode='rgb', classes=None, 
                    class_mode='binary', batch_size=batch_size, shuffle=False, seed=500, 
                    save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                    interpolation='nearest', drop_duplicates=False)
    
    test_previous_generator = test_previous_datagen.flow_from_dataframe(test_df, directory=test_dir, x_col='filename_housing', y_col=regression_columns, 
                    target_size=(224, 224), color_mode='rgb', classes=None, 
                    class_mode='other', batch_size=batch_size, shuffle=False, seed=500, 
                    save_to_dir=None, save_prefix='', save_format='png', subset=None, 
                    interpolation='nearest', drop_duplicates=False)
    while True:
      x_test = test_generator.next()
      x_previous_test = test_previous_generator.next()
      yield [x_test[0] ], [x_previous_test[1], x_test[1]]
 

# Now we proceed to decapitate the model, and retrain its head on the new data.

# In[8]:


from keras.applications.xception import Xception

#xception_weights_path = "/home/leila/Code/siamese_traffic_density/pretrained/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
xception_conv_base = Xception(include_top=False, weights="imagenet", input_tensor=None, input_shape=(224, 224, 3), pooling=None, classes=None)
xception_conv_base.summary()
#xception_conv_base.load_weights(xception_weights_path)


# In[9]:


from keras.models import Model
from keras.layers import Flatten, Dense, Input

current_input = Input(shape= (224, 224, 3), name="current_input")

xception_model_embeddings = xception_conv_base (current_input)
flattened = Flatten(xception_model_embeddings)
dense_1 = Dense(1024, activation='relu') (flattened)
dense_2 = Dense(512, activation='relu') (dense_1)
dense_3 = Dense(256, activation='relu') (dense_2)

## Output
class_pred = Dense(3, activation='softmax', name = "final_pred_class") (dense_3)
reg_pred = Dense(2, name = "final_pred_reg") (dense_3)

xception_model = Model([current_input], [class_pred, reg_pred ])

losses = {
    'final_pred_reg': 'mae',
    'final_pred_class': 'categorical_crossentropy'
}
xception_model.compile(optimizer="adagrad",loss=losses, metrics=["accuracy"])


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
from math import ceil

training_steps = ceil(4826/32)
validation_steps = ceil(604/32)
pre_file_path = "weights-improvement-"
post_file_path = "--{epoch:02d}-{val_loss:.2f}.hdf5"
filepath= pre_file_path + post_file_path

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
callbacks_list = [stopping, checkpoint]


xception_history = xception_model.fit_generator(train_gen(), steps_per_epoch=training_steps, epochs=50, verbose=1, callbacks=callbacks_list, validation_data=val_gen(),
                             validation_steps = validation_steps,class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, 
                             shuffle=False)


# In[ ]:


import matplotlib.pyplot as plt
loss = xcpetion_history.history['loss']
val_loss = xcpetion_history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


index = train_df['similarity'].index[train_df['similarity'].apply(np.isnan)]


# In[ ]:


index


# In[ ]:




