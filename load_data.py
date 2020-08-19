import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import glob
import cv2 as cv

def load_data():
"""
load the image data as 4D array, resize to 224 x 224 and normalize it
apply one-hot transformation to the label
split the data into training and validation set in 8:2
"""
	train_val_path = Path('datasets/chest_xray/trainval')

	nor_path = train_val_path/'NORMAL'
	pne_path = train_val_path/'PNEUMONIA'
	nor_case = nor_path.glob('*.jpeg')
	pne_case = pne_path.glob('*.jpeg')

	train_val_data = []
	train_val_label = []

	for img in nor_case:
		img = cv.imread(str(img))
		img = cv.resize(img, (224,224))
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(float)/255
		label = to_categorical(0, num_classes=2)
		train_val_data.append(img)
		train_val_label.append(label)

	for img in pne_case:
		img = cv.imread(str(img))
		img = cv.resize(img, (224,224))
		img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(float)/255
		label = to_categorical(1, num_classes=2)
		train_val_data.append(img)
		train_val_label.append(label)

	train_val_data = np.array(train_val_data)
	train_val_label = np.array(train_val_label)
	
	x_train, x_val, y_train, y_val = train_test_split(train_val_data, train_val_label, test_size = 0.2, shuffle = True)

	return x_train, x_val, y_train, y_val
