import tensorflow as tf
import os
import datetime
 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D,Activation,Dropout,Dense,Flatten
from tensorflow.keras.optimizers import Adam
from keras import backend as k
from keras import layers
from tensorflow.keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
 
 
import cv2
import matplotlib.pyplot as plt

import json
import os

sPath = "D:/UNI/MAESTRIA/2DO CICLO/TOPICO IA/Proyecto Final/"
imagePaths = os.listdir(sPath+"test")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(sPath+'ModelNames.xml')

try:
	print("LEYENDO MODELO")
	json_file = open(sPath+"model.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights(sPath+"model.h5")
	model.compile(optimizer=Adam(lr=0.00002),loss='categorical_crossentropy',metrics=['accuracy'])
except IOError:
	print("CREANDO MODELO")
	batch_size = 32 #32,128,256
	epochs = 20 #10
	 
	train_data_dir = sPath+"train"
	test_data_dir  = sPath+"test"
	 
	# Preparing data
	trainGen = ImageDataGenerator(rescale=1./255,shear_range=0.2,horizontal_flip=True,zoom_range=0.2)
	testGen = ImageDataGenerator(rescale=1./255)
	train = trainGen.flow_from_directory(train_data_dir,target_size=(224,224),classes=['with_mask','without_mask'],class_mode = 'categorical',batch_size=batch_size,shuffle=True)
	test = testGen.flow_from_directory(test_data_dir,target_size=(224,224),classes=['with_mask','without_mask'],class_mode = 'categorical',batch_size=batch_size)
	 
	mob = MobileNetV2(alpha=1.3,
		input_shape = (224,224,3),
		include_top = False,
		weights = 'imagenet',
	)
	mob.trainable = False
	 
	model = Sequential()
	model.add(mob)
	model.add(GlobalAveragePooling2D())
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(2,activation='softmax'))
	model.summary()
	model.compile(optimizer=Adam(lr=0.00002),loss='categorical_crossentropy',metrics=['accuracy'])
	model.fit(train,epochs=epochs,validation_data=test)	
	
	model_json = model.to_json()
	with open(sPath+"model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights(sPath+"model.h5")

def predict_mask(path):
    im = cv2.imread(path)
    im_resized = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
 
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    plt.show()
     
    img_pred = image.load_img(path,target_size=(224,224))
    img_pred = image.img_to_array(img_pred)
    img = np.expand_dims(img_pred,axis=0)
    result = model.predict_classes(img)
    prob = model.predict_proba(img)
    print(result)
    print('Probability:{}'.format(prob))
    if result[0]==0:
        prediction ="MSK"
    else:
        prediction ="No MSK"
 
    print(prediction)
 
 
# Function calling 
 
#predict_mask(sPath+"test/with_mask/JFP.jpg")
#predict_mask(sPath+"test/without_mask/JFP.jpeg")
 
 
 
# messages on screen
resMap = {
        0 : 'Con Mascarilla',
        1 : 'Sin Mascarilla'
    }
 
# Colors on screen
colorMap = {
        0 : (0,255,0),
        1 : (0,0,255)
    }
 
def prepImg(pth):
    return cv2.resize(pth,(224,224)).reshape(1,224,224,3)/255.0

def prepImgName(pth):
    return cv2.resize(pth,(150,150),interpolation=cv2.INTER_CUBIC)
 
classifier = cv2.CascadeClassifier(sPath+"haarcascade_frontalface_default.xml")
 
cap = cv2.VideoCapture(0)
auxcont = 0
while True:

    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1.1,2)
    auxFrame = gray.copy()
    for face in faces:
        auxcont+=1
        rostro = auxFrame[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
        #slicedImg = img[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
#check if has a mask
        pred = model.predict(prepImg(img))
        (mask, womask) = pred[0]
        pred = np.argmax(pred)
        label = "{}: {:.2f}%".format(resMap[pred], max(mask, womask) * 100)
#check who is
        result = face_recognizer.predict(prepImgName(rostro))
        person_name = imagePaths[result[0]] if result[1]>50 else 'Desconocido'
        name = "{}: {:.2f}%".format(person_name,result[1] if result[1] < 100 else 100) #'{}'.format(result) #if result[1] < 50 else 'desconocido'
#print result
        cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),colorMap[pred],2)
        cv2.putText(img,label,(face[0],face[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,colorMap[pred],2)
        cv2.putText(img,name,(face[0],face[3]+100),cv2.FONT_HERSHEY_SIMPLEX,0.5,colorMap[pred],2)
        #print('Segundo: '+str(auxcont)+'=> '+label)
        #print(label)
        #print('Segundo: '+str(auxcont)+'=> '+name)
    cv2.imshow('SISTEMA DETECCION DE ROSTROS Y MASCARILLAS POR CAMARA',img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
     
cap.release()
cv2.destroyAllWindows()