
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def equalize_class_size(Data_in):
    Data=Data_in.copy()
    class_num=np.min(Data[Data.shape[1]-1].value_counts())
    classes=(Data[Data.shape[1]-1].value_counts().index)

    frames=[]
    for i in classes:
        Data2=Data.copy()
        filter = Data2[Data2.shape[1]-1]==i
        Data2.where(filter , inplace=True)
        Data2=Data2.dropna()
        Data2=Data2.sample(n=class_num,replace=False,random_state=0)
        frames.append(Data2)
    Data=pd.concat(frames)
    return Data


def make_train_test_Data(features,label,pca_num,train_ratio):
    Data=pd.DataFrame(np.concatenate([features,label],axis=1))
    x =  Data.loc[:,[i for i in range(features.shape[1])]].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=pca_num)
    principalComponents = pca.fit_transform(x)
    Data=pd.DataFrame(np.concatenate([principalComponents,label],axis=1))
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