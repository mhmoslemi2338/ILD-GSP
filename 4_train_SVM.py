import os
import scipy.io
import numpy as np
import pandas as pd
from functions_SVM import *

try: os.mkdir('results')
except: pass
#*******************************************************************
#************************** Prepare Data ***************************
#*******************************************************************

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


######## Talisman ##########
label_Talisman=[]
features_Talisman=[]

for ll in labels_name:
    label_path=os.path.join('features/Test_Talisman',ll)
    files=os.listdir(label_path)
    try: files.remove('.DS_Store')
    except: pass
    for row in files:
        file_path=os.path.join(label_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_Talisman.append(feature_vector[0])

    label_Talisman+=[labels_dict[ll]]*len(files)

features_Talisman=np.array(features_Talisman)
label_Talisman=np.array(label_Talisman)
label_Talisman = np.reshape(label_Talisman,(features_Talisman.shape[0],1))


# ####### statistics #####
Data_ILD=pd.DataFrame(np.concatenate([features_ILD,label_ILD],axis=1))
Data_Talisman=pd.DataFrame(np.concatenate([features_Talisman,label_Talisman],axis=1))
# print('ILD class Distribution:')
# tmp=pd.DataFrame(Data_ILD[features_ILD.shape[1]].value_counts())
# tmp=tmp.rename(index=dict((v,k) for k,v in labels_dict.items()))
# tmp=tmp.rename(columns={48:'count'})
# print(tmp)
# print('\nTalisman Distribution Dist:')
# tmp=pd.DataFrame(Data_Talisman[features_Talisman.shape[1]].value_counts())
# tmp=tmp.rename(index=dict((v,k) for k,v in labels_dict.items()))
# tmp=tmp.rename(columns={48:'count'})
# print(tmp)



#**************************************************************
#************************** Trainng ***************************
#**************************************************************
ratio=0.75
pca_num=30

[ILD_train,ILD_test]=make_train_test_Data(features_ILD,label_ILD,pca_num,ratio)
[CM_ILD,acc_ILD]=train_SVM(ILD_train,ILD_test) 
print('\nAccuracy on ILD, RBF kernel,',pca_num,'PCA components: ',round(acc_ILD,3),'%')

[Talisman_train,Talisman_test]=make_train_test_Data(features_Talisman,label_Talisman,pca_num,ratio)
[CM_Talisman,acc_Talisman]=train_SVM(Talisman_train,Talisman_test) 
print('Accuracy on Talisman, RBF kernel,',pca_num,'PCA components: ',round(acc_Talisman,3),'%\n')

###### plot confusion matrix ######
plot_CM(CM_ILD,labels_name,'Confusion Matrix for ILD Data',True)
plot_CM(CM_Talisman,labels_name,'Confusion Matrix for Talisman Data',True)

##### acc values
stat_ILD=statistics_CM(CM_ILD,labels_name,labels_dict)
stat_Talisman=statistics_CM(CM_Talisman,labels_name,labels_dict)
with open('results/ILD.txt', mode='w') as file_object:
    print(stat_ILD, file=file_object)
with open('results/Talisman.txt', mode='w') as file_object:
    print(stat_Talisman, file=file_object)


#************************** SAME CLASS SIZE ***************************
Data_ILD_equ = equalize_class_size(Data_ILD)
Data_Talisman_equ = equalize_class_size(Data_Talisman)
label_ILD = np.array(Data_ILD_equ[48]).reshape(-1,1)
features_ILD = Data_ILD_equ.drop(columns=48)
label_Talisman = np.array(Data_Talisman_equ[48]).reshape(-1,1)
features_Talisman = Data_Talisman_equ.drop(columns=48)

[ILD_train,ILD_test]=make_train_test_Data(features_ILD,label_ILD,pca_num,ratio)
[CM_ILD,acc_ILD]=train_SVM(ILD_train,ILD_test) 
print('\nAccuracy on ILD, RBF kernel,',pca_num,'PCA components: ',round(acc_ILD,3),'%')

[Talisman_train,Talisman_test]=make_train_test_Data(features_Talisman,label_Talisman,pca_num,ratio)
[CM_Talisman,acc_Talisman]=train_SVM(Talisman_train,Talisman_test) 
print('Accuracy on Talisman, RBF kernel,',pca_num,'PCA components: ',round(acc_Talisman,3),'%\n')

###### plot confusion matrix ######
plot_CM(CM_ILD,labels_name,'Confusion Matrix for ILD Data-same class size',True)
plot_CM(CM_Talisman,labels_name,'Confusion Matrix for Talisman Data-same class size',True)

##### acc values
stat_ILD=statistics_CM(CM_ILD,labels_name,labels_dict)
stat_Talisman=statistics_CM(CM_Talisman,labels_name,labels_dict)
with open('results/ILD-same class size.txt', mode='w') as file_object:
    print(stat_ILD, file=file_object)
with open('results/Talisman-same class size.txt', mode='w') as file_object:
    print(stat_Talisman, file=file_object)

