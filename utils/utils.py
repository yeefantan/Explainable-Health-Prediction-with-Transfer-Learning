import os
import numpy as np
import cv2
from glob import glob
from matplotlib import pyplot
from sklearn.utils import shuffle
import tensorflow as tf
from PIL import Image
import imageio
import glob
import random
from scipy import ndimage

def load_data(datadir,categories):
    
    datalength = 0
    data = list()
    labels = list()
    for i,category in enumerate(categories):
        path = os.path.join(datadir, category)
        path_list = os.listdir(path)
        if ('.DS_Store') in path_list:
            path_list.remove('.DS_Store')
        for img in path_list:
            img_ = os.path.join(path,img)
            img_ = cv2.imread(img_)
            img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
            data.append(img_)

    return np.asarray(data)

def get_label_augmented_data(symptom,normal):
    """return train data and label"""
    data_normal = list()
    labels = list()
    data_symptoms = list()
    for image in normal:
        image2 = tf.image.flip_left_right(image)
        data_normal.append(np.asarray(image))
        data_normal.append(np.asarray(image2))
        data_normal.append(np.asarray(tf.image.rot90(image,tf.random.uniform(shape=[],
                                                                              minval=1,maxval=4,
                                                                             dtype=tf.int32))))
        data_normal.append(np.asarray(tf.image.random_brightness(image,max_delta=0.3)))
        image3 = ndimage.rotate(image,15)
        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(image,-1,kernel)
        data_normal.append(image3)
        data_normal.append(dst)

    for image in symptom:
        image2 = tf.image.flip_left_right(image)
        data_symptoms.append(np.asarray(image))
        data_symptoms.append(np.asarray(image2))
        data_symptoms.append(np.asarray(tf.image.rot90(image,tf.random.uniform(shape=[],
                                                                              minval=1,maxval=4,
                                                                             dtype=tf.int32))))
        data_symptoms.append(np.asarray(tf.image.random_brightness(image,max_delta=0.3)))
        image3 = ndimage.rotate(image,15)

        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(image,-1,kernel)
        data_symptoms.append(image3)
        data_symptoms.append(dst)
        
    train_data = np.concatenate((np.array(data_normal),
                       np.array(data_symptoms)))
    labels = np.concatenate((np.zeros(6000),np.ones(3600)))
        
    return train_data,labels


def haar_crop():

    path_symptoms = "Train/Symptoms_ori"
    path_normal = "Train/Normal_ori"

    savepath_symptoms = "Train/Symptoms_crop/"
    savepath_normal = "Train/Normal_crop/"

    facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    found_faces = list()

    dirs = os.listdir(path_symptoms)

    if('.DS_Store' in dirs):
        dirs.remove('.DS_Store')

    i = 0

    for item in dirs:
        if os.path.isfile(path_symptoms+'/'+item):
            image = cv2.imread(path_symptoms+'/'+item)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces = facecascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30,30))
            if(len(faces)==1):
                found_faces.append(item)
                for (x,y,w,h) in faces:
                    cv2.rectangle(image,(x,y),(x+w, y+h), (0,255,0),2)
                    roi_color = image[y+2:y+h-2, x+2:x+w-2]
                    cv2.imwrite(savepath_symptoms+item, roi_color)


    found_faces = list()

    dirs = os.listdir(path_normal)

    if('.DS_Store' in dirs):
        dirs.remove('.DS_Store')

    i = 0

    for item in dirs:
        if os.path.isfile(path_normal+'/'+item):
            image = cv2.imread(path_normal+'/'+item)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces = facecascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30,30))
            if(len(faces)==1):
                found_faces.append(item)
                for (x,y,w,h) in faces:
                    cv2.rectangle(image,(x,y),(x+w, y+h), (0,255,0),2)
                    roi_color = image[y+2:y+h-2, x+2:x+w-2]
                    cv2.imwrite(savepath_normal+item, roi_color)




def resize_rename():
	path_symptoms = "Train/Symptoms_crop/"
	path_normal = "Train/Normal_crop/"
	save_path = "Train/Normal/"
	i = 0
	dirs = os.listdir(path_symptoms)
	dirs.remove('.DS_Store')
	for item in dirs:
	    if os.path.isfile(path_symptoms+item):
	        image = Image.open(path_symptoms+item)
	        image_resized = image.resize((200,200))
	        image_resized.save(save_path+str(i)+'_symptoms.jpg','PNG', quality=100)
	        i+=1

	save_path = "Train/Symptoms/"
	i = 0
	dirs = os.listdir(path_normal)
	dirs.remove('.DS_Store')
	for item in dirs:
	    if os.path.isfile(path_normal+item):
	        image = Image.open(path_normal+item)
	        image_resized = image.resize((200,200))
	        image_resized.save(save_path+str(i)+'_normal.jpg','PNG', quality=100)
	        i+=1
    
