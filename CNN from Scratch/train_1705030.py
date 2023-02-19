
import os
import cv2
import numpy as np
import pandas as pd
import pickle

# global_images = []


def read_images_from_folder(folder_path):
    images = []
    image_size = 32
    for filename in os.listdir(folder_path):
        # print(filename)
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(cv2.resize(img,(image_size,image_size),interpolation=cv2.INTER_LINEAR))
    return np.array(images)

def process_data(load_test = False):
    train_percent = .9
    # print("==============================================================================")
    folder_path = r"S:\STUDY\ML Sessional CSE472\ofl9 4\Code\training-a"
    test_folder_path = r"S:\STUDY\ML Sessional CSE472\ofl9 4\Code\training-d"
    csv_file = r"S:\STUDY\ML Sessional CSE472\ofl9 4\Code\training-a.csv"
    test_csv_file = r"S:\STUDY\ML Sessional CSE472\ofl9 4\Code\training-d.csv"
    images = read_images_from_folder(folder_path)
    images=255-images
    images=np.where(images<80,0,1)
    # print(images.shape)
    

    train_size=int(images.shape[0]*train_percent)
    
    
    train_X = images[:train_size]
    validation_X = images[train_size:]
    
    df = pd.read_csv(csv_file)
    digits = df["digit"]
    train_Y = digits[:train_size]
    validation_Y = digits[train_size:]
    
    
    test_X = None
    test_Y = None
    
    if(load_test):
        test_X = read_images_from_folder(test_folder_path)
        test_X=255-test_X
        test_X=np.where(test_X<80,0,1)
        
        test_df = pd.read_csv(test_csv_file)
        
        test_digits = test_df["digit"]

        test_Y = test_digits
    # print(images.shape)
    # print(digits.shape)
    
    return train_X, validation_X, test_X, train_Y, validation_Y, test_Y


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math

import time
import numpy as np
np.random.seed(6)

import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics


class Convolution_Layer:
        
    def __init__(self,stride,padding,kernel_count,kernel_shape,input_channel_count,lr):
        self.lr = lr
        self.padding=padding
        
       
        self.kernel_shape=kernel_shape
        self.stride=stride
        


        self.kernel=np.random.randn(kernel_shape,kernel_shape,input_channel_count,kernel_count)*np.sqrt(2/(self.kernel_shape*self.kernel_shape*input_channel_count))
    
        self.bias=np.zeros((1,1,1,kernel_count))
        self.forward_input = None


    def forward_pass(self,input):

        input = np.pad(input, ((0, 0), (self.padding,self.padding),(self.padding,self.padding),(0,0)))
        self.forward_input=input
        
        h=int(np.floor(input.shape[1]-self.kernel.shape[0]+self.stride[0]/self.stride[0]))
        w=int(np.floor(input.shape[2]-self.kernel.shape[1]+self.stride[1]/self.stride[1]))
        expanded_input = np.lib.stride_tricks.as_strided(
        input,
        shape=(
            self.kernel.shape[0],
            self.kernel.shape[1],
            self.kernel.shape[2],
            input.shape[0],
            h,
            w,
            1
        ),
        strides=(
            (input.strides*2)[1:]
        ),
        writeable=False,  
    )

 
        return (np.einsum(
        'pqrs,pqrtbmn->tbms',
        self.kernel,
        expanded_input,
    )+self.bias)
        
    def backward_pass(self,dL):
        
        modified_shape=(dL.shape[1]-1)*self.stride[0]+1
        
        dL_dil=np.zeros((dL.shape[0],modified_shape,modified_shape,self.bias.shape[3]))
        dL_dil[:,:: self.stride[0],::self.stride[0],:]=dL
    
        
   

        input_modified = np.lib.stride_tricks.as_strided(
        self.forward_input,
        shape=(

            self.forward_input.shape[1]-dL_dil.shape[1]+1,
            self.forward_input.shape[2]-dL_dil.shape[2]+1,
            dL.shape[3],
            self.forward_input.shape[3],
            dL.shape[1],
            dL.shape[2]
        ),
        strides=(
            (self.forward_input.strides*2)[2:]
        ),
        writeable=False,  
    )

        
        dw= np.einsum(
        'sbmr,pqrtbm->pqtr',
        dL_dil,
        input_modified,
    )
        
        
        
        rotated_kernel = np.rot90(np.transpose(self.kernel, (0, 1, 3, 2)), 2, axes=(0, 1))
        dL_pad =np.pad(dL_dil,((0,),(self.kernel_shape-self.padding-1,),(self.kernel_shape-self.padding-1,),(0,)))
        
        dL_pad_modified = np.lib.stride_tricks.as_strided(
        dL_pad,
        shape=(
            
            rotated_kernel.shape[0],
            rotated_kernel.shape[1],
            rotated_kernel.shape[2],
            dL_pad.shape[0],
            dL_pad.shape[1]-rotated_kernel.shape[0]+1,
            dL_pad.shape[2]-rotated_kernel.shape[1]+1
        ),
        strides=(
            (dL_pad.strides*2)[2:]
            
        ),
        writeable=False,  
    )
        
        dx= np.einsum(
        'pqrtbm, pqrs->tbms',
        
        dL_pad_modified,
        rotated_kernel
    )
        
        db=np.reshape((np.sum(dL,axis=(0,1,2))/dL.shape[0]),(1,1,1,self.bias.shape[3]))
        
        self.kernel=self.kernel-self.lr*dw
        self.bias=self.bias-self.lr*db
        
        self.forward_input = None
        return dx
    
    def destructor(self):
        self.forward_input = None
        
        
class MaxPool_Layer:
    def __init__(self,pool_shape,stride):
        self.pool_shape=pool_shape
        self.stride=stride
        self.forward_input_shape = None
        self.mask = None

    def forward_pass(self,input):
        self.forward_input_shape=input.shape
    

        height_ = int(math.floor((input.shape[1]-self.pool_shape) // self.stride[0] + 1))
        width_ = int(math.floor((input.shape[2]-self.pool_shape) // self.stride[1] + 1))
        
        
        batch_s,height_s,width_s,channel_s=input.strides[0], input.strides[1]*self.stride[0], input.strides[2]*self.stride[1], input.strides[3]
        modified_input=np.lib.stride_tricks.as_strided(
            input,
            shape=(
                input.shape[0],
                height_,
                width_,
                input.shape[3],
                self.pool_shape,
                self.pool_shape),           
            strides=(
                
                batch_s,
                height_s,
                width_s,
                channel_s,
                int(height_s/self.stride[0]),
                int(width_s/self.stride[1])
                
                ))
        
        X=np.max(modified_input,axis=(4,5))
        
        modified_X=X.repeat(self.pool_shape,axis=1)
        modified_X=modified_X.repeat(self.pool_shape,axis=2)
        
        modified_input=input[:,:height_*self.stride[0],:width_*self.stride[1],:]
        
        self.mask=np.where(modified_input==modified_X, 1, 0)
 
        
        return X
    
    def backward_pass(self,dL):
        
         modified_dL=dL.repeat(self.pool_shape,axis=1)
         modified_dL=modified_dL.repeat(self.pool_shape,axis=2)
         
         modified_dL=modified_dL*self.mask
         self.mask = None
         
         X = np.zeros(self.forward_input_shape)
         X[:,:modified_dL.shape[1],:modified_dL.shape[2],:]=modified_dL
         return X
     
    def destructor(self):
        self.mask = None

                        
#DONE
class Relu_Layer:
    def __init__(self):
        
        self.backward_mul = None
    def forward_pass(self,input):
        
        self.backward_mul=np.where(input>0,1,0)
        return np.where(input>0,input,0)
          
    def backward_pass(self,dL):
       
        
        dk = dL*self.backward_mul
        self.backward_mul = None
        return dk
    def destructor(self):
        self.backward_mul = None
    
    
#DONE    
class Softmax_Layer:
    def forward_pass(self,input):
        return np.exp(input.T)/np.sum(np.exp(input.T),axis=0)
    def backward_pass(self,dL):
        return dL
    
#DONE
class Flattening_layer:
    def __init__(self):
        self.pre_shape = None
    def forward_pass(self, input):
        self.pre_shape = input.shape
        return input.flatten('C').reshape(input.shape[0],-1)
    def backward_pass(self, input):
        return input.reshape(self.pre_shape)
    
        
class Fully_Connected_Layer:
    def __init__(self,in_shape,out_shape, lr):
        self.forward_input=None
        self.lr = lr 
        self.weight = None
        self.bias = None
        self.update_parameters(0,0,in_shape,out_shape,0)
        
    def update_parameters(self, dw, db, in_shape, out_shape, mode):
        if(mode==0):
            self.bias=np.zeros((out_shape,1))
            self.weight=np.random.randn(out_shape,in_shape)*(np.sqrt(2/in_shape))
            return
        self.weight= self.weight - dw*self.lr
        self.bias = self.bias - db*self.lr


    def forward_pass(self,input):
        self.forward_input=input
        return (np.dot(self.weight,input.T) + self.bias).T
    

    
    def backward_pass(self,dL):
        
        dx = np.dot(dL,self.weight)
        dw=np.dot(dL.T,self.forward_input)
        db=np.sum(dL.T,axis=1,keepdims=True)
        self.update_parameters(dw, db, 0, 0, 1)
        self.forward_input=None

        return dx
    def destructor(self):
        self.forward_input=None
    

        

def cross_entropy_loss(pred_y,true_y):
    
    return np.sum( -np.log ( pred_y ) * true_y)
    

class Lenet_Model:
    def __init__(self, input_x, input_y, batchsize, epoch, lr):
        self.lr = lr
        self.batchsize = batchsize
        self.batch_count = int(input_x.shape[0]/self.batchsize)
        self.epoch = epoch
        self.input_x=input_x
        self.input_y=input_y
        
        self.conv1=Convolution_Layer(np.array([1,1]),0,6,5,input_x.shape[3],self.lr)
        self.rel1=Relu_Layer()
        self.max_pool1=MaxPool_Layer(2,np.array([2,2]))
        
        self.conv2=Convolution_Layer(np.array([1,1]),0,16,5,6,self.lr)
        self.rel2=Relu_Layer()
        self.max_pool2=MaxPool_Layer(2,np.array([2,2]))
        
        self.flat = Flattening_layer()
        
        self.fc1 = Fully_Connected_Layer(400, 120, lr)
        self.fc2 = Fully_Connected_Layer(120, 84, lr)
        self.fc3 = Fully_Connected_Layer(84, 10, lr)
        
        self.soft_max = Softmax_Layer()
        
    def forward_pass(self, X):
        X=self.conv1.forward_pass(X)
        X=self.rel1.forward_pass(X)
        X=self.max_pool1.forward_pass(X)
        
        X=self.conv2.forward_pass(X)
        X=self.rel2.forward_pass(X)
        X=self.max_pool2.forward_pass(X)
        
        X=self.flat.forward_pass(X)
        
        X=self.fc1.forward_pass(X)
        X=self.fc2.forward_pass(X)
        X=self.fc3.forward_pass(X)
        
        X=self.soft_max.forward_pass(X)
        
        return X
        
    def backward_pass(self, X):
    
        X=self.soft_max.backward_pass(X)
        X=self.fc3.backward_pass(X)
        X=self.fc2.backward_pass(X)
        X=self.fc1.backward_pass(X)
        X=self.flat.backward_pass(X)
        X=self.max_pool2.backward_pass(X)
        X=self.rel2.backward_pass(X)
        X=self.conv2.backward_pass(X)
        X=self.max_pool1.backward_pass(X)
        X=self.rel1.backward_pass(X)
        X=self.conv1.backward_pass(X)
        
        return X
    
    def F1_score(self, train_y, X):
        pred_y = np.argmax(X,axis=0)   
        return f1_score(train_y, pred_y, average='macro')
    
    def accurecy_score(self, train_y, X):
        pred_y = np.argmax(X,axis=0)  
        return np.sum(np.where(pred_y==train_y, 1, 0))
            
        
    def train(self):
        
        t_acc = []
        t_loss = []
        t_f1 = []
        v_acc = []
        v_loss = []
        v_f1 = []
       
        
        for epoch_count in range(self.epoch):
            print("Epoch No: : ",epoch_count+1)

            sum_loss=0
           
            sum_accuracy=0
            sum_f1=0
            

            for batch_count in tqdm(range(self.batch_count), desc='Epoch'):

                train_x = self.input_x[batch_count*self.batchsize:batch_count*self.batchsize+self.batchsize]
                train_y = self.input_y[batch_count*self.batchsize:batch_count*self.batchsize+self.batchsize]
                
                X = self.forward_pass(train_x)
                    
                
                
                accuracy = accuracy_score(train_y, np.argmax(X,axis=0))
                
                
            
                y_one_hot = np.zeros(( X.shape[0], train_y.shape[0])) 
                y_one_hot[train_y, np.arange(train_y.shape[0])] = 1
                
                loss=cross_entropy_loss(X,y_one_hot)
                f1 = self.F1_score(train_y, X)
                

   
                dL=(X-y_one_hot).T/self.batchsize
                
                
                
                
                self.backward_pass(dL)
                
                sum_loss=sum_loss + loss
    
                sum_accuracy=sum_accuracy+accuracy
                sum_f1=sum_f1+f1
                
                
               
                    

            sum_loss=(sum_loss/self.batch_count)/self.batchsize
            
            
            sum_accuracy=(sum_accuracy/self.batch_count)*100
            sum_f1 = (sum_f1/self.batch_count)
            
            t_acc.append(sum_accuracy)
            t_loss.append(sum_loss)
            t_f1.append(sum_f1)
            
            valid_acc, valid_f1, valid_loss =self.validation(validation_X, validation_Y)
            
            v_acc.append(valid_acc)
            v_loss.append(valid_loss)
            v_f1.append(valid_f1)
      
            print("Training: \n","Loss: ",sum_loss,"Accuracy: ",sum_accuracy,"F1 score: ",sum_f1)          
            print("Validation: \n","Loss: ",valid_loss,"Accuracy: ",valid_acc,"F1 score: ",valid_f1)
            print("\n\n")
        self.input_x=None
        self.input_y=None
        self.plot_figure(t_acc, v_acc, "Training accuracy", "Validation accuracy", "Accuracy")
        self.plot_figure(t_loss, v_loss, "Training loss", "Validation loss", "Loss")
        self.plot_figure(t_f1, v_f1, "Training f1", "Validation f1", "F1 Score")
        

    def plot_figure(self, train, valid, label_t, label_v, lab):
        epo = range(1,len(train)+1)
        plt.plot(epo, train, 'g', label=label_t)
        plt.plot(epo, valid, 'b', label=label_v)
        # plt.title(lab)
        plt.xlabel('Epochs')
        plt.ylabel(lab)
        plt.legend()
        plt.show()
    
    def confusion_plot(self,true_y, pred_y):
        confusion_matrix = metrics.confusion_matrix(true_y, pred_y)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0,1,2,3,4,5,6,7,8,9])

        cm_display.plot()
        
        plt.show()
        
            

    def test(self, test_X, test_Y):
        print("Testing...")
        pred_y = self.forward_pass(test_X)
        accuracy = accuracy_score(test_Y, np.argmax(pred_y,axis=0) )
        f1 = self.F1_score(test_Y, pred_y)
        
        y_one_hot = np.zeros(( pred_y.shape[0], test_Y.shape[0])) 
        y_one_hot[test_Y, np.arange(test_Y.shape[0])] = 1
        
        loss=cross_entropy_loss(pred_y,y_one_hot)/len(test_Y)
        return accuracy*100, f1, loss, np.argmax(pred_y,axis=0) 
    
    def only_test(self, test_X):
        print("Testing...")
        pred_y = self.forward_pass(test_X)
        return np.argmax(pred_y,axis=0) 
        
    def validation(self, valid_X, valid_Y):
        print("Validating...")
        
        pred_y = self.forward_pass(valid_X)
        accuracy = accuracy_score(valid_Y, np.argmax(pred_y,axis=0) )
        f1 = self.F1_score(valid_Y, pred_y)
        
        y_one_hot = np.zeros(( pred_y.shape[0], valid_Y.shape[0])) 
        y_one_hot[valid_Y, np.arange(valid_Y.shape[0])] = 1
        
        loss=cross_entropy_loss(pred_y,y_one_hot)/len(valid_Y)
        # self.confusion_plot(valid_Y, np.argmax(pred_y,axis=0))
        
        return accuracy*100, f1, loss
    
    def destructor(self):
        self.conv1.destructor()
        self.rel1.destructor()
        self.max_pool1.destructor()
        
        self.conv2.destructor()
        self.rel2.destructor()
        self.max_pool2.destructor()
        
        # self.flat.destructor()
        
        self.fc1.destructor()
        self.fc2.destructor()
        self.fc3.destructor()
        
        # self.soft_max.destructor()
    
    







if __name__ == "__main__":
    # process_data()

    training_X, validation_X, test_X, training_Y, validation_Y, test_Y = process_data(load_test=False)
    print(training_X.shape)
    print(validation_X.shape)
    batchsize = 32
    epoch = 8
    lr = 0.01


    lenet = Lenet_Model(training_X, training_Y, batchsize, epoch, lr)
    lenet.train()
    lenet.destructor()
    # lenet.test(test_X, test_Y)
    print("DONE")

    
    with open("1705030_model.pickle", "wb") as f:
        pickle.dump(lenet, f)