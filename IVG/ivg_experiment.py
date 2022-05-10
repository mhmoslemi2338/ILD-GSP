
import numpy as np
import pandas as pd
import os
import scipy.io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
from sklearn import svm

try: os.mkdir('results')
except: pass

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



def plot_CM(CM_in,labels_name,name,is_save):
    CM=CM_in.copy()
    CM=CM/CM.sum(axis=1)[:,None]
    (a,a)=CM.shape
    ###### save confussion matrix as an image #####
    fig=plt.figure(figsize=(16, 14))
    plt.imshow(CM,  cmap=plt.cm.Blues);
    thresh=CM.max()/2
    for i in range(a):
        for j in range(a):
            number=100*CM[i, j].copy()
            if number==float(0):
                number=int(number)
            number=round(number,2)
            if(CM[i, j] > thresh) : color="white"
            else: color="black"
            fontweight='normal'
            fontsize=20
            if float(number)>50 : 
                fontsize=22
            plt.text(j, i,number ,horizontalalignment="center",color=color,fontsize=fontsize)
    plt.xticks(np.arange(a), labels_name,fontsize='x-large',rotation=-30,fontweight='bold')
    plt.yticks(np.arange(a),  labels_name,fontsize='x-large',fontweight='bold')
    plt.title(name,fontsize=20,fontweight='bold'); plt.ylabel('True label',fontsize=20); plt.xlabel('Predicted label',fontsize=20);
    if is_save:
        fig.savefig('results/'+name+'.jpg', dpi=3*fig.dpi)
        plt.close(fig)




def statistics_CM(CM_in,labels_name,labels_dict):
    cm=CM_in.copy()
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    num_classes = len(labels_name)
    TN = []
    for i in range(num_classes):
        temp = np.delete(cm, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))

    acc=TP/np.sum(cm, axis=1)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F1_score=2*precision*recall/(precision+recall)

    acc=np.append(acc,np.average(acc,weights=np.sum(CM_in,axis=1)/np.sum(CM_in)))
    precision=np.append(precision,np.average(precision,weights=np.sum(CM_in,axis=1)/np.sum(CM_in)))
    recall = np.append(recall,np.average(recall,weights=np.sum(CM_in,axis=1)/np.sum(CM_in)))
    specificity=np.append(specificity,np.average(specificity,weights=np.sum(CM_in,axis=1)/np.sum(CM_in)))
    F1_score=np.append(F1_score,np.average(F1_score,weights=np.sum(CM_in,axis=1)/np.sum(CM_in)))


    df=pd.DataFrame(
        {"Accuracy":np.round(100*acc,2),
        "Recall":np.round(100*recall,2),
        "Precision":np.round(100*precision,2),
        "Specificity":np.round(100*specificity,2),
        "F1-score":np.round(100*F1_score,2)})
    tmp=dict((v-1,k) for k,v in labels_dict.items())
    tmp[5]='All classes'
    return df.rename(index=tmp)



#****************************************************************
#****************************************************************
#****************************************************************



######## labels name ##########
feature_name='HVG_I_lattice'
labels_name=os.listdir(feature_name+'/Train_ILD')
try: labels_name.remove('.DS_Store')
except: pass
labels_dict={}
for i in range(len(labels_name)):
    labels_dict[labels_name[i]]=i+1

######## extract DATA ##########

### HVG 2I lattice features
feature_name='HVG_2I_lattice'
label_ILD=[]
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
    label_ILD+=[labels_dict[ll]]*len(files)
features_HVG_2I_lattice=np.array(features_HVG_2I_lattice)
label_ILD=np.array(label_ILD)
label_ILD = np.reshape(label_ILD,(features_HVG_2I_lattice.shape[0],1))


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


###### Training #####
Data=pd.DataFrame(np.concatenate([features_HVG_2I_lattice, features_HVG_2I_Nolattice,features_IVG_2I_lattice,features_IVG_2I_Nolattice,features_wavelet,label_ILD],axis=1))
Train=Data.groupby(Data.shape[1]-1).apply(lambda s: s.sample(frac=0.75,replace=False,random_state=0))
Train = Train.reset_index(level=[None])
Train=Train.set_index('level_1')
Test= Data.drop(index=Train.index)


x_train =  Train.loc[:,[i for i in range(Train.shape[1]-1)]].values
y_train = Train.loc[:,[Train.shape[1]-1]].values.ravel()
x_train, y_train=unison_shuffled_copies(x_train, y_train)

x_test =  Test.loc[:,[i for i in range(Test.shape[1]-1)]].values
y_test = Test.loc[:,[Test.shape[1]-1]].values.ravel()
x_test, y_test=unison_shuffled_copies(x_test, y_test)


####### data preprocessing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

pca = PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
pca_num = np.argmax(cumsum > 0.95)
pca = PCA(n_components=pca_num)

x_train = pca.fit_transform(x_train)
x_test=pca.transform(x_test)



########### TRAINING ##############
clf = svm.SVC(kernel='rbf',decision_function_shape='ovo'  , class_weight='balanced'  ,max_iter=-1)
clf.fit(x_train, y_train)

#### calc result on Test
label_predict_val=[]
for row in x_test:
    ll=clf.predict([row])
    label_predict_val.append(ll)
label_predict_val=np.array(label_predict_val)
label_predict_val=label_predict_val.reshape(y_test.shape)
diff=(label_predict_val==y_test)
acc=100*np.sum(diff)/len(y_test) 
CM=confusion_matrix(y_test, label_predict_val)
TP = np.diag(CM)
FP = np.sum(CM, axis=0) - TP
FN = np.sum(CM, axis=1) - TP
precision=TP/(TP+FP)

recall=TP/(TP+FN)
F1_score=100*np.average(2*precision*recall/(recall+precision))
# print('\n***** ',names[i],'*****')
print('Feature vector length:',Train.shape[1])
print('\nSVM, RBF kernel,',pca_num,'PCA components: ')
print('\tAccuracy on Test: ' ,round(acc,3),'%')
print('\tF1_score on Test: ',round(F1_score,3),'%')


###### plot confusion matrix ######
plot_CM(CM,labels_name,'Confusion Matrix for ILD Test Data',True)
stat_ILD=statistics_CM(CM,labels_name,labels_dict)
with open('results/ILD_Test.txt', mode='w') as file_object:
    print(stat_ILD, file=file_object)



#### calc result on Train
label_predict_train=[]
for row in x_train:
    ll=clf.predict([row])
    label_predict_train.append(ll)
label_predict_train=np.array(label_predict_train)
label_predict_train=label_predict_train.reshape(y_train.shape)
diff=(label_predict_train==y_train)
acc=100*np.sum(diff)/len(y_train) 
CM=confusion_matrix(y_train, label_predict_train)
TP = np.diag(CM)
FP = np.sum(CM, axis=0) - TP
FN = np.sum(CM, axis=1) - TP
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1_score=100*np.average(2*precision*recall/(recall+precision))
print('SVM, RBF kernel,',pca_num,'PCA components: ')
print('\tAccuracy on Train: ' ,round(acc,3),'%')
print('\tF1_score on Train: ',round(F1_score,3),'%')


###### plot confusion matrix ######
plot_CM(CM,labels_name,'Confusion Matrix for ILD Train Data',True)
stat_ILD=statistics_CM(CM,labels_name,labels_dict)
with open('results/ILD_Train.txt', mode='w') as file_object:
    print(stat_ILD, file=file_object)
