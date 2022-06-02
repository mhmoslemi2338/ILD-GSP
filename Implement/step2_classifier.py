import os
from pycm import *
import scipy.io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold 

try: os.mkdir("result")
except: pass

labels_name=['healthy', 'ground', 'micronodules', 'emphysema', 'fibrosis']
labels_dict={'healthy': 1, 'ground': 2, 'micronodules': 3, 'emphysema': 4, 'fibrosis': 5}

def plot_CM(CM,labels_name,name,is_save):
    fig=plt.figure(figsize=(12, 12))
    plt.imshow(CM, cmap=plt.cm.Blues);
    for i in range(CM.shape[0]):
        for j in range(CM.shape[0]):
            if(CM[i, j] > CM.max()/2) : color="white"
            else: color="black"
            plt.text(j, i,CM[i, j] ,horizontalalignment="center",color=color,fontsize=17)
    plt.xticks(np.arange(CM.shape[0]), labels_name,fontsize='x-large',rotation=-30,fontweight='bold')
    plt.yticks(np.arange(CM.shape[0]),  labels_name,fontsize='x-large',fontweight='bold')
    plt.title(name,fontsize=18,fontweight='bold'); plt.ylabel('True label',fontsize=18); plt.xlabel('Predicted label',fontsize=18);
    if is_save:
        fig.savefig("result/"+name+'.jpg', dpi=3*fig.dpi)
        plt.close(fig)



#------ Read Feature vectors ------
feature_name='texture_features'
label_ILD=[]
features=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector'] 
        features.append(feature_vector[0])
    label_ILD+=[labels_dict[ll]]*len(files)
features=np.array(features)
label_ILD=np.array(label_ILD)
label_ILD = np.reshape(label_ILD,(features.shape[0],1))

#------ Prepare Data ------
Data=pd.DataFrame(np.concatenate([features, label_ILD],axis=1))
Train=Data.sample(frac=1,random_state=5) ## shuffle
x_train =  Train.loc[:,[i for i in range(Train.shape[1]-1)]].values
y_train = Train.loc[:,[Train.shape[1]-1]].values.ravel()
y_test_all=np.array([])
y_prediction_all=np.array([])


skfolds = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
for train_index, test_index in skfolds.split(x_train, y_train):
    #------ Train , Test folds ------
    x_train_folds = x_train[train_index]
    y_train_folds = y_train[train_index]
    x_test_fold = x_train[test_index]
    y_test_fold = y_train[test_index]

    #------ Data preprocess ------
    scaler = StandardScaler()
    pca = PCA()
    x_train_folds = scaler.fit_transform(x_train_folds)
    pca.fit(x_train_folds)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    pca_num = np.argmax(cumsum > 0.95)
    pca = PCA(n_components=pca_num)
    x_train_folds = pca.fit_transform(x_train_folds)

    #------ Training ------
    clf = svm.SVC(kernel='rbf',decision_function_shape='ovo' ,class_weight='balanced'  ,max_iter=-1)
    clf.fit(x_train_folds, y_train_folds)

    x_test_fold=scaler.transform(x_test_fold)
    x_test_fold=pca.transform(x_test_fold)
    label_predict_test=clf.predict(x_test_fold)

    y_test_all=np.concatenate([y_test_all,y_test_fold])
    y_prediction_all=np.concatenate([y_prediction_all,label_predict_test])
    

#------ result ------
cm = ConfusionMatrix(actual_vector=y_test_all, predict_vector=y_prediction_all)
CM=np.array([list(row.values()) for row in list(cm.matrix.values())])
plot_CM(CM,labels_name,'CM_ILD_Test',True)

accuracy=np.array(list((cm.ACC).values()))
precision=np.array(list((cm.PPV).values()))
recall=np.array(list((cm.TPR).values()))
true_negative_rate=np.array(list((cm.TNR).values()))
AUC=np.array(list((cm.AUC).values()))
F1=np.array(list((cm.F1).values()))

overall_accuracy=str(round(100*cm.Overall_ACC,2))
overal_precision=str(round(100*cm.PPV_Macro,2))
overal_recall=str(round(100*cm.TPR_Macro,2))
overal_true_negative_rate=str(round(100*cm.TNR_Macro,2))
overal_AUC='-'
overal_F1=str(round(100*cm.F1_Macro,2))

overal=[overall_accuracy , overal_precision , overal_recall 
        , overal_true_negative_rate , overal_AUC , overal_F1]


df=pd.DataFrame(
    {"Accuracy":np.round(100*accuracy,2),
    "Recall":np.round(100*recall,2),
    "Precision":np.round(100*precision,2),
    "TN rate":np.round(100*true_negative_rate,2),
    "AUC":np.round(100*AUC,2),
    "F1-score":np.round(100*F1,2)})
df=df.rename(index=dict((v-1,k) for k,v in labels_dict.items()))
df.loc['Overal']=overal
with open('result/ILD_Test.txt', mode='w') as file_object:
    print(df, file=file_object)

print('\nFeature vector length:',Train.shape[1]-1)
print('SVM, RBF kernel,PCA components: ',pca_num)
print('\tAccuracy on Test: ' ,overall_accuracy,'%')
print('\tF1_score on Test: ',overal_F1,'%')


##### LATEX result format
CM_label=['H','GG','M','E','F']
with open('result/CM_latex.txt', mode='w') as file_object:
    for j,row in enumerate(CM):
        print(CM_label[j]+' & ',end='', file=file_object)
        for i,row2 in enumerate(row):
            if i!=len(row)-1:
                print(str(row2)+' &',end=' ', file=file_object)
            else:
                if j!=len(row)-1:
                    print(str(row2)+" \\\ ", file=file_object)
                else:
                    print(str(row2)+" \\\ \\bottomrule", file=file_object)
######
with open('result/ILD_Test.txt') as f:
    data=f.readlines()
for i in range(len(data)):
    data[i]=data[i].strip().split()
data=data[1:]
data[5][0]=data[5][0] + ' & '
for i in range(6):
    if i!=5:
        data[i][0]=CM_label[i].split()[0] + ' & '
    for j in range(5):
        if i==5 and j==4 : 
            data[i][j+1]=data[i][j+1]+' & '
            continue
        data[i][j+1]=data[i][j+1]+'\% & '
    data[i][6]=data[i][6]+ '\% \\\ '
data[-2][-1]=data[-2][-1]+"  \\bottomrule"
data.append(['\\vspace{-0.3cm} \\\ '])
with open('result/acc_latex.txt', mode='w') as file_object:
    for row in data[:-2]:
        print("".join(row),file=file_object)
    print("\\vspace{-0.25cm} \\\ ",file=file_object)
    print("".join(data[-2]),file=file_object)
    print("".join(data[-1]),file=file_object)
    print("\\bottomrule",file=file_object)

