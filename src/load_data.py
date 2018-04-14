import os
import cv2
import numpy as np
import h5py
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def load_image(path):
    pathList, jpgList, labelList, imageList = [], [], [], []

    for i, folder in enumerate(os.listdir(path)):
        sub_path = os.path.join(path, folder)
        print('{}:Processing {}'.format(i, folder))
        
        if os.path.isdir(sub_path):
            for image in os.listdir(sub_path):
                image = os.path.join(sub_path, image)
                if image.endswith('.jpg'):
                    img = cv2.resize(cv2.imread(image),(64,64))  #讀取圖片與縮小
                    labelList.append(i)  #將Label存起
                    pathList.append(image)  #將資料夾名稱存起
                    imageList.append(img)  #將圖片存起
    
    return labelList, pathList, imageList

def get_read_main(path, save = False, load = False):
    if load:  #讀取 h5 檔
        h5f = h5py.File('dataset.h5','r')  
        X_train = h5f['X_train'][:]
        X_test = h5f['X_valid'][:]
        h5f.close()    

        h5f = h5py.File('labels.h5','r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_valid'][:]
        h5f.close()  
    
    else:
        labelList, pathList, imageList = load_image(path)
        img_arr = np.array(imageList)
        enc = OneHotEncoder()
        label_arr = np.array(labelList).reshape(-1, 1)
        labels = enc.fit_transform(label_arr).toarray() 
        
        X_train, X_test, y_train, y_test = train_test_split(img_arr, labels, test_size=0.3)
        
        if save:
            h5f = h5py.File('dataset.h5', 'w')   
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_valid', data=X_test)
            h5f.close()

            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_valid', data=y_test)
            h5f.close()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255   
    
    return X_train, X_test, y_train, y_test