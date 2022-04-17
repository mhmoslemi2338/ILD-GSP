



import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tensorflow.keras import models, layers 
import tensorflow as tf

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def my_convert(arr_in):
    tmp=np.zeros((arr_in.shape[0],5))
    for i,row in enumerate(arr_in):
        tmp[i][int(row)-1]=1 
    return tmp.copy()


######## labels name ##########
labels_name=os.listdir('features/Train_ILD')
try: labels_name.remove('.DS_Store')
except: pass
labels_dict={}
for i in range(len(labels_name)):
    labels_dict[labels_name[i]]=i+1

######## ILD ##########
label_ILD=[]
features_ILD=[]

for ll in labels_name:
    label_path=os.path.join('features/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_ILD.append(feature_vector[0])

    label_ILD+=[labels_dict[ll]]*len(files)

features_ILD=np.array(features_ILD)
label_ILD=np.array(label_ILD)
label_ILD = np.reshape(label_ILD,(features_ILD.shape[0],1))



# ***** PCA *********
Data=pd.DataFrame(np.concatenate([features_ILD,label_ILD],axis=1))
x =  Data.loc[:,[i for i in range(features_ILD.shape[1])]].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=30)
principalComponents = pca.fit_transform(x)
Data=pd.DataFrame(np.concatenate([principalComponents,label_ILD],axis=1))



# ***** Prepare Data *********
Train1=Data.groupby(Data.shape[1]-1).apply(lambda s: s.sample(frac=0.9,replace=False,random_state=0))
Train1 = Train1.reset_index(level=[None])
Train1=Train1.set_index('level_1')
Test= Data.drop(index=Train1.index)
Train=Train1.groupby(Data.shape[1]-1).apply(lambda s: s.sample(frac=0.72,replace=False,random_state=0))
Train = Train.reset_index(level=['level_1'])
Train=Train.set_index('level_1')
Validation= Train1.drop(index=Train.index)


x_train =  Train.loc[:,[i for i in range(Train.shape[1]-1)]].values
y_train = Train.loc[:,[Train.shape[1]-1]].values.ravel()
x_train, y_train=unison_shuffled_copies(x_train, y_train)

x_test =  Test.loc[:,[i for i in range(Test.shape[1]-1)]].values
y_test = Test.loc[:,[Test.shape[1]-1]].values.ravel()
x_test, y_test=unison_shuffled_copies(x_test, y_test)

x_val =  Validation.loc[:,[i for i in range(Validation.shape[1]-1)]].values
y_val = Validation.loc[:,[Validation.shape[1]-1]].values.ravel()
x_val, y_val=unison_shuffled_copies(x_val, y_val)


# ********** Train neural network  **********
epochs_num=60
with tf.device('/CPU:0'):
    ### layer input
    inputs = layers.Input(x_train.shape[1:])
    ### hidden layers
    hidden1 = layers.Dense(80,name="hidden1",activation='relu')(inputs)
    drop1=layers.Dropout(0.4)(hidden1)
    hidden2 = layers.Dense(80,name="hidden2", activation='relu')(drop1)
    drop2=layers.Dropout(0.4)(hidden2)
    hidden3 = layers.Dense(15,name="hidden3", activation='relu')(drop2)
    ### layer output
    outputs = layers.Dense(5,name="output", activation='softmax')(hidden3)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, my_convert(y_train), validation_data=(x_val,my_convert(y_val)),epochs=epochs_num, verbose=1)

    Test_loss, Test_acc = model.evaluate(x_test, my_convert(y_test), verbose=1)
    print('\nTest loss:',str(round(Test_loss,5)))
    print('Test Accuracy:',str(round(100*Test_acc,4)),' %\n')



fig=plt.figure(figsize=(15,12))
plt.subplot(211)
plt.plot(range(1, epochs_num+1), np.array(history.history['accuracy'])*100, '*--', label='Training acc')
plt.plot(range(1, epochs_num+1), np.array(history.history['val_accuracy'])*100, '*--', label='Validation acc')
plt.title('Training and Validation Accuracy, acc for Test:'+str(round(100*Test_acc,4))+'%',fontsize='x-large'); 
plt.legend();plt.grid();plt.xlabel("Epoch",fontsize='x-large'); plt.ylabel("Acuuracy",fontsize='x-large')
plt.subplot(212)
plt.plot(range(1, epochs_num+1), history.history['loss'], '*--', label='Training loss')
plt.plot(range(1, epochs_num+1), history.history['val_loss'], '*--', label='Validation loss')
plt.title('Training and Validation loss',fontsize='x-large')
plt.grid();plt.xlabel("Epoch",fontsize='x-large'); plt.ylabel("Loss",fontsize='x-large'); plt.legend()
# plt.show()
fig.savefig('results/NN_train.jpg', dpi=3*fig.dpi)
fig.close()
