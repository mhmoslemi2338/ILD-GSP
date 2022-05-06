
import numpy as np
import pandas as pd
import os
import scipy.io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  confusion_matrix
from sklearn import svm

try: os.mkdir('results')
except: pass

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]





######## labels name ##########
feature_name='HVG_I_lattice'
labels_name=os.listdir(feature_name+'/Train_ILD')
try: labels_name.remove('.DS_Store')
except: pass
labels_dict={}
for i in range(len(labels_name)):
    labels_dict[labels_name[i]]=i+1

######## extract DATA ##########

### HVG I lattice features
feature_name='HVG_I_lattice'
label_ILD=[]
features_HVG_I_lattice=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_HVG_I_lattice.append(feature_vector[0])
    label_ILD+=[labels_dict[ll]]*len(files)
features_HVG_I_lattice=np.array(features_HVG_I_lattice)
label_ILD=np.array(label_ILD)
label_ILD = np.reshape(label_ILD,(features_HVG_I_lattice.shape[0],1))

### HVG 2I lattice features
feature_name='HVG_2I_lattice'
features_HVG_2I_lattice=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_HVG_2I_lattice.append(feature_vector[0])
features_HVG_2I_lattice=np.array(features_HVG_2I_lattice)

### HVG I Nolattice features
feature_name='HVG_I_Nolattice'
features_HVG_I_Nolattice=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_HVG_I_Nolattice.append(feature_vector[0])
features_HVG_I_Nolattice=np.array(features_HVG_I_Nolattice)

### HVG 2I Nolattice features
feature_name='HVG_2I_Nolattice'
features_HVG_2I_Nolattice=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_HVG_2I_Nolattice.append(feature_vector[0])
features_HVG_2I_Nolattice=np.array(features_HVG_2I_Nolattice)

### IVG I lattice features
feature_name='IVG_I_lattice'
features_IVG_I_lattice=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_IVG_I_lattice.append(feature_vector[0])
features_IVG_I_lattice=np.array(features_IVG_I_lattice)

### IVG 2I lattice features
feature_name='IVG_2I_lattice'
features_IVG_2I_lattice=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_IVG_2I_lattice.append(feature_vector[0])
features_IVG_2I_lattice=np.array(features_IVG_2I_lattice)

### IVG I Nolattice features
feature_name='IVG_I_Nolattice'
features_IVG_I_Nolattice=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_IVG_I_Nolattice.append(feature_vector[0])
features_IVG_I_Nolattice=np.array(features_IVG_I_Nolattice)

### IVG 2I Nolattice features
feature_name='IVG_2I_Nolattice'
features_IVG_2I_Nolattice=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_IVG_2I_Nolattice.append(feature_vector[0])
features_IVG_2I_Nolattice=np.array(features_IVG_2I_Nolattice)

### wavelet features
features_wavelet=[]
for ll in labels_name:
    label_path=os.path.join('/Users/mohammad/Desktop/Bsc prj/Graph wavelet implement/features/Train_ILD',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_wavelet.append(feature_vector[0])
features_wavelet=np.array(features_wavelet)




features=[
    [features_HVG_2I_lattice,features_IVG_2I_Nolattice],
    [features_HVG_2I_Nolattice,features_IVG_2I_lattice]]


names=['features_HVG_lattice_IVG_Nolattice_symmetric','features_HVG_Nolattice_IVG_lattice_symmetric']

for i in range(len(features)):
    # ***** Prepare Data *********
    Data=pd.DataFrame(np.concatenate([features[i][0], features[i][1],label_ILD],axis=1))
    Train1=Data.groupby(Data.shape[1]-1).apply(lambda s: s.sample(frac=0.99994,replace=False,random_state=0))
    Train1 = Train1.reset_index(level=[None])
    Train1=Train1.set_index('level_1')
    Test= Data.drop(index=Train1.index)
    Train=Train1.groupby(Data.shape[1]-1).apply(lambda s: s.sample(frac=0.75,replace=False,random_state=0))
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

    # print('\n',x_train.shape, x_val.shape, x_test.shape)

    ####### data preprocessing
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val=scaler.transform(x_val)
    x_test=scaler.transform(x_test)

    pca = PCA()
    pca.fit(x_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    pca_num = np.argmax(cumsum > 0.95)
    pca = PCA(n_components=pca_num)

    x_train = pca.fit_transform(x_train)
    x_val=pca.transform(x_val)
    x_test=pca.transform(x_test)



    ########### TRAINING ##############
    clf = svm.SVC(kernel='rbf',decision_function_shape='ovo'  , class_weight='balanced'  ,max_iter=-1)
    clf.fit(x_train, y_train)

    #### calc result on Test
    label_predict_val=[]
    for row in x_val:
        ll=clf.predict([row])
        label_predict_val.append(ll)
    label_predict_val=np.array(label_predict_val)
    label_predict_val=label_predict_val.reshape(y_val.shape)
    diff=(label_predict_val==y_val)
    acc=100*np.sum(diff)/len(y_val) 
    CM=confusion_matrix(y_val, label_predict_val)
    TP = np.diag(CM)
    FP = np.sum(CM, axis=0) - TP
    FN = np.sum(CM, axis=1) - TP
    precision=TP/(TP+FP)
    
    recall=TP/(TP+FN)
    F1_score=100*np.average(2*precision*recall/(recall+precision))
    print('\n***** ',names[i],'*****')
    print('Feature vector length:',Train1.shape[1])
    print('\nSVM, RBF kernel,',pca_num,'PCA components: ')
    print('\tAccuracy on Test: ' ,round(acc,3),'%')
    print('\tF1_score on Test: ',round(F1_score,3),'%')


    # #### calc result on Train
    # label_predict_train=[]
    # for row in x_train:
    #     ll=clf.predict([row])
    #     label_predict_train.append(ll)
    # label_predict_train=np.array(label_predict_train)
    # label_predict_train=label_predict_train.reshape(y_train.shape)
    # diff=(label_predict_train==y_train)
    # acc=100*np.sum(diff)/len(y_train) 
    # CM=confusion_matrix(y_train, label_predict_train)
    # TP = np.diag(CM)
    # FP = np.sum(CM, axis=0) - TP
    # FN = np.sum(CM, axis=1) - TP
    # precision=TP/(TP+FP)
    # recall=TP/(TP+FN)
    # F1_score=100*np.average(2*precision*recall/(recall+precision))
    # print('SVM, RBF kernel,',pca_num,'PCA components: ')
    # print('\tAccuracy on Train: ' ,round(acc,3),'%')
    # print('\tF1_score on Train: ',round(F1_score,3),'%')
