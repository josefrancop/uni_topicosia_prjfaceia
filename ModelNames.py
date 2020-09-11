import cv2
import os
import numpy as np 

dataPath = "D:/UNI/MAESTRIA/2DO CICLO/TOPICO IA/Proyecto Final/test"
peopleList = os.listdir(dataPath)
print('Lista de Personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:

	personaPath = dataPath + '/' + nameDir
	print('Leyendo las imagenes')

	for fileName in os.listdir(personaPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personaPath + '/' + fileName,0))
		image = cv2.imread(personaPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1		

#print('labels= ',labels)
#print('numero de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#cv2.destroyAllWindows()

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Entrenando el reconocedor de rostros

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigenFace.xml')
face_recognizer.write('ModelNames.xml')

print("Modelo Almacenando....")