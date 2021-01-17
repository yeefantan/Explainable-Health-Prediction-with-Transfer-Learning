import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils.utils import *
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D,GlobalAveragePooling2D,Dense,Conv2D, Convolution2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tqdm
import os
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from keras import applications
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import keras
import sklearn.metrics as metrics
import seaborn as sns
from keras_vggface.vggface import VGGFace

def create_features(dataset,model):
    
    train_data = np.expand_dims(dataset,axis=0)
    train_data = np.vstack(train_data)
    features = model.predict(train_data)
    features_flatten = features.reshape((features.shape[0],6*6*512))
    
    return train_data, features, features_flatten


def model():
	data_normal = load_data('Augmentation Data',['Normal'])
	data_symptoms = load_data('Augmentation Data',['Symptoms'])
	train_data = np.concatenate((np.array(data_normal),
                           np.array(data_symptoms)))
	labels = np.concatenate((np.zeros(6000),np.ones(3600)))

	model = VGGFace(weights='vggface',
                        include_top=False,
                        input_shape=(200,200,3))

	train_data, features, features_flatten = create_features(train_data,model)

	X_train, X_valid, y_train,y_valid = train_test_split(features,labels,test_size=0.2,random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.15,random_state=42)

	checkpointer = ModelCheckpoint(filepath='model/cnn_best.hdf5',verbose=1,
                              save_best_only=True)

	es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
	model = Sequential()
	model.add(Conv2D(32, (3,3), activation='relu', input_shape=(6, 6, 512))) 
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu',padding="Same")) 
	model.add(MaxPooling2D((2, 2)))



	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(2, activation='softmax'))

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
	              metrics=['accuracy'])

	model_history = model.fit(X_train,y_train,
                         epochs=100,
                         callbacks=[es,checkpointer],
                         verbose=2,
                         validation_data=(X_valid,y_valid))

	return model


def evaluate(model,X_test,y_test):
	print(model.evaluate(X_test,y_test))
	predictions_ = model.predict(X_test)
	predictions = np.argmax(predictions_, axis=1)

	print(classification_report(y_test,predictions))
	print("Accuracy:",accuracy_score(y_test,predictions))


def load_model():
	model = tf.keras.models.load_model('model/cnn_best.hdf5')
	return model

