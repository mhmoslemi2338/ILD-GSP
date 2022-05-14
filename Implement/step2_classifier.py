import os
from pycm import *
import scipy.io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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




### read features and concert to dataframe
feature_name='texture_features/graph'
label_ILD=[]
features_graph=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector'] 
        features_graph.append(feature_vector[0])
    label_ILD+=[labels_dict[ll]]*len(files)
features_graph=np.array(features_graph)
label_ILD=np.array(label_ILD)
label_ILD = np.reshape(label_ILD,(features_graph.shape[0],1))

feature_name='texture_features/wavelet'
features_wavelet=[]
for ll in labels_name:
    label_path=os.path.join(feature_name+'/',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_wavelet.append(feature_vector[0])
features_wavelet=np.array(features_wavelet)



for same_class_size in [False , True]:
    if same_class_size: name="_same_class_size"
    else: name=""

    ########### select Train and Test sets #############
    # split data between train and test 
    # we choose 25% of data for Test
    # after selecting Train , test we shuffles each set using unison_shuffled_copies
    ####################################################
    Data=pd.DataFrame(np.concatenate([features_graph,features_wavelet, label_ILD],axis=1))
    if same_class_size:
        class_size=np.min(Data[Data.shape[1]-1].value_counts())
        Data=Data.groupby(Data.shape[1]-1).apply(lambda s: s.sample(n=class_size,replace=False,random_state=0))
        Data = Data.reset_index(level=[None])
        Data=Data.set_index('level_1')

    Data=Data.sample(frac=1,random_state=5) ## shuffle
    Train=Data.sample(frac=0.75,replace=False,random_state=0)
    Test= Data.drop(index=Train.index)

    x_train =  Train.loc[:,[i for i in range(Train.shape[1]-1)]].values
    y_train = Train.loc[:,[Train.shape[1]-1]].values.ravel()

    x_test =  Test.loc[:,[i for i in range(Test.shape[1]-1)]].values
    y_test = Test.loc[:,[Test.shape[1]-1]].values.ravel()

    ########### Data preprocessing ##############
    #  Data preprocessing is:
    #       1) zero-mean and scale variances to one 
    #       2) PCA for 0.95% of total varince
    #############################################
    scaler = StandardScaler()
    pca = PCA()
    x_train = scaler.fit_transform(x_train)
    pca.fit(x_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    pca_num = np.argmax(cumsum > 0.95)
    pca = PCA(n_components=pca_num)
    x_train = pca.fit_transform(x_train)


    ########### TRAINING ##############
    clf = svm.SVC(kernel='rbf',decision_function_shape='ovo' ,class_weight='balanced'  ,max_iter=-1)
    clf.fit(x_train, y_train)

    #**** predict label for Test data
    x_test=scaler.transform(x_test)
    x_test=pca.transform(x_test)
    label_predict_test=clf.predict(x_test)

    #*****
    cm = ConfusionMatrix(actual_vector=y_test, predict_vector=label_predict_test)
    CM=np.array([list(row.values()) for row in list(cm.matrix.values())])
    plot_CM(CM,labels_name,'CM_ILD_Test'+name,True)

    #*****
    accuracy=np.array(list((cm.ACC).values()))
    precision=np.array(list((cm.PPV).values()))
    recall=np.array(list((cm.TPR).values()))
    true_negative_rate=np.array(list((cm.TNR).values()))
    AUC=np.array(list((cm.AUC).values()))
    F1=np.array(list((cm.F1).values()))
    overall_accuracy=round(100*cm.Overall_ACC,2)
    overal_F1=round(100*cm.F1_Macro,2)

    df=pd.DataFrame(
        {"Accuracy":np.round(100*accuracy,2),
        "Recall":np.round(100*recall,2),
        "Precision":np.round(100*precision,2),
        "TN rate":np.round(100*true_negative_rate,2),
        "AUC":np.round(100*AUC,2),
        "F1":np.round(100*F1,2)})
    df=df.rename(index=dict((v-1,k) for k,v in labels_dict.items()))
    with open('result/ILD_Test'+name+'.txt', mode='w') as file_object:
        print(df, file=file_object)
        print('\n\t**** overal accuracy: '+str(overall_accuracy)+' % ', file=file_object)
        print('\t**** overal F1-score (Macro): '+str(overal_F1)+' %', file=file_object)
    #*****
    print('\nFeature vector length'+name+' :',Train.shape[1])
    print('SVM, RBF kernel,PCA components: ',pca_num)
    print('\tAccuracy on Test: ' ,overall_accuracy,'%')
    print('\tF1_score on Test: ',overal_F1,'%')

