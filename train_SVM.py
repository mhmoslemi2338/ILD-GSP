

import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def make_train_test_Data(features,lable,pca_num,train_ratio):
    Data=pd.DataFrame(np.concatenate([features,lable],axis=1))
    x =  Data.loc[:,[i for i in range(features.shape[1])]].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=pca_num)
    principalComponents = pca.fit_transform(x)
    Data=pd.DataFrame(np.concatenate([principalComponents,lable],axis=1))
    # make train_test
    train=Data.sample(frac=train_ratio,replace=False,random_state=0)
    test=Data.drop(train.index)
    return [train,test]

def train_SVM(train, test):
    x_train =  train.loc[:,[i for i in range(train.shape[1]-1)]].values
    y_train = train.loc[:,[train.shape[1]-1]].values.ravel()

    x_test=  test.loc[:,[i for i in range(test.shape[1]-1)]].values
    y_test = test.loc[:,[test.shape[1]-1]].values.ravel()

    clf = svm.SVC(kernel='rbf',decision_function_shape='ovo'  , class_weight='balanced'  ,max_iter=-1)
    clf.fit(x_train, y_train)

    ####### ACC on Test  ########
    label_predict_Test=[]
    for row in x_test:
        ll=clf.predict([row])
        label_predict_Test.append(ll)

    label_predict_Test=np.array(label_predict_Test)
    label_predict_Test=label_predict_Test.reshape(y_test.shape)
    diff=(label_predict_Test==y_test)
    acc=100*np.sum(diff)/len(y_test) 
    CM=confusion_matrix(y_test, label_predict_Test)
    return [CM,acc]

def plot_CM(CM_in,name,is_save):
    CM=CM_in.copy()
    CM=CM/CM.sum(axis=1)[:,None]
    (a,a)=CM.shape
    ###### save confussion matrix as an image #####
    fig=plt.figure(figsize=(16, 14))
    plt.imshow(CM,  cmap=plt.cm.Blues);
    thresh=CM.max()/2
    for i in range(a):
        for j in range(a):
            number=CM[i, j].copy()
            if number==float(0):
                number=int(number)
            number=round(number,3)
            if(CM[i, j] > thresh) : color="white"
            else: color="black"
            fontweight='normal'
            fontsize=20
            if float(number)>0.5 : 
                fontsize=22
            plt.text(j, i,number ,horizontalalignment="center",color=color,fontsize=fontsize)
    plt.xticks(np.arange(a), lables_name,fontsize='x-large',rotation=-30,fontweight='bold')
    plt.yticks(np.arange(a),  lables_name,fontsize='x-large',fontweight='bold')
    plt.title(name,fontsize=20,fontweight='bold'); plt.ylabel('True label',fontsize=20); plt.xlabel('Predicted label',fontsize=20);
    if is_save:
        fig.savefig(name+'.jpg', dpi=3*fig.dpi)
        plt.close(fig)




def statistics_CM(CM_in):
    cm=CM_in.copy()
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    num_classes = len(lables_name)
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
    tmp=dict((v-1,k) for k,v in lables_dict.items())
    tmp[5]='All classes'
    return df.rename(index=tmp)


#*******************************************************************
#************************** Prepare Data ***************************
#*******************************************************************


######## Lables name ##########
lables_name=os.listdir('features/Train_ILD')
try: lables_name.remove('.DS_Store')
except: pass
lables_dict={}
for i in range(len(lables_name)):
    lables_dict[lables_name[i]]=i+1

######## ILD ##########
lable_ILD=[]
features_ILD=[]

for ll in lables_name:
    lable_path=os.path.join('features/Train_ILD',ll)
    files=os.listdir(lable_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(lable_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_ILD.append(feature_vector[0])

    lable_ILD+=[lables_dict[ll]]*len(files)

features_ILD=np.array(features_ILD)
lable_ILD=np.array(lable_ILD)
lable_ILD = np.reshape(lable_ILD,(features_ILD.shape[0],1))



######## Talisman ##########
lable_Talisman=[]
features_Talisman=[]

for ll in lables_name:
    lable_path=os.path.join('features/Test_Talisman',ll)
    files=os.listdir(lable_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(lable_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_Talisman.append(feature_vector[0])

    lable_Talisman+=[lables_dict[ll]]*len(files)

features_Talisman=np.array(features_Talisman)
lable_Talisman=np.array(lable_Talisman)
lable_Talisman = np.reshape(lable_Talisman,(features_Talisman.shape[0],1))


####### statistics #####
Data_ILD=pd.DataFrame(np.concatenate([features_ILD,lable_ILD],axis=1))
Data_Talisman=pd.DataFrame(np.concatenate([features_Talisman,lable_Talisman],axis=1))
print('ILD class Distribution:')
tmp=pd.DataFrame(Data_ILD[features_ILD.shape[1]].value_counts())
tmp=tmp.rename(index=dict((v,k) for k,v in lables_dict.items()))
tmp=tmp.rename(columns={48:'count'})
print(tmp)
print('\nTalisman Distribution Dist:')
tmp=pd.DataFrame(Data_Talisman[features_Talisman.shape[1]].value_counts())
tmp=tmp.rename(index=dict((v,k) for k,v in lables_dict.items()))
tmp=tmp.rename(columns={48:'count'})
print(tmp)



#**************************************************************
#************************** Trainng ***************************
#**************************************************************


ratio=0.75
pca_num=30


[ILD_train,ILD_test]=make_train_test_Data(features_ILD,lable_ILD,pca_num,ratio)
[CM_ILD,acc_ILD]=train_SVM(ILD_train,ILD_test) 
print('\nAccuracy on ILD, RBF kernel,',pca_num,'PCA components: ',round(acc_ILD,3),'%')

[Talisman_train,Talisman_test]=make_train_test_Data(features_Talisman,lable_Talisman,pca_num,ratio)
[CM_Talisman,acc_Talisman]=train_SVM(Talisman_train,Talisman_test) 
print('Accuracy on Talisman, RBF kernel,',pca_num,'PCA components: ',round(acc_Talisman,3),'%\n')


###### plot confusion matrix ######
plot_CM(CM_ILD,'Confusion Matrix for ILD Data',True)
plot_CM(CM_Talisman,'Confusion Matrix for Talisman Data',True)




##### acc values
stat_ILD=statistics_CM(CM_ILD)
stat_Talisman=statistics_CM(CM_Talisman)

with open('ILD.txt', mode='w') as file_object:
    print(stat_ILD, file=file_object)

with open('Talisman.txt', mode='w') as file_object:
    print(stat_Talisman, file=file_object)
