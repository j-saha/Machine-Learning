from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import csv
import sys
import os
import cv2
import numpy as np
import pandas as pd
from train_1705030 import *
import pickle
# global_images = []


def read_images_from_folder(folder_path):
    images = []
    file_list = []
    image_size = 32
    for filename in os.listdir(folder_path):
        if ("png" in filename):
            file_list.append(filename)
            img = cv2.imread(os.path.join(folder_path,filename))
            if img is not None:
                images.append(cv2.resize(img,(image_size,image_size),interpolation=cv2.INTER_LINEAR))
    return np.array(images), file_list

def process_test_data(test_folder_path, csv=False):
    print("Data reading...")
    # test_folder_path = r"S:\STUDY\ML Sessional CSE472\ofl9 4\Code\training-d"
    test_csv_file = r"S:\STUDY\ML Sessional CSE472\ofl9 4\Code\training-d.csv"

    test_X, file_list = read_images_from_folder(test_folder_path)
    test_X=255-test_X
    test_X=np.where(test_X<90,0,1)
    
    test_Y = None
    if(csv):

        test_df = pd.read_csv(test_csv_file)

        test_digits = test_df["digit"]

        test_Y = test_digits
    # print(images.shape)
    # print(digits.shape)

    return test_X, test_Y, file_list




def F1_score(train_y, X):
    pred_y = np.argmax(X,axis=0)   
    return f1_score(train_y, pred_y, average='macro')

def cross_entropy_loss(pred_y,true_y):
    return np.sum( -np.log ( pred_y ) * true_y)






if __name__ == "__main__":
    iscsv = False
    path = sys.argv[1]
    print(path)
    test_X, test_Y, file_list = process_test_data(path, iscsv)
    with open("1705030_model.pickle", "rb") as f:
        model = pickle.load(f)
        
        if(iscsv):
            test_acc, test_f1, test_loss, pred_y = model.test(test_X, test_Y)
            model.confusion_plot(test_Y, pred_y)
            print("Test: \n","Loss: ",test_loss,"Accuracy: ",test_acc,"F1 score: ",test_f1)
            print("\n\n")
        else:
            pred_y = model.only_test(test_X)
        
        with open(str(path)+str("\\1705030_prediction.csv"), 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(['FileName', 'Digit'])

            # Write the data rows
            # print(len(file_list))
            # print(pred_y.shape)
            for i in range(len(file_list)):
                writer.writerow([file_list[i], pred_y[i]])
        
        
    
