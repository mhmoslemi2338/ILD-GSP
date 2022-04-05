

import os
import scipy.io
import numpy as np
from sklearn import svm
import random


ration=0.75

######## ILD ##########
lable_ILD=[]
features_ILD=[]

lable_ILD_train=[]
features_ILD_train=[]

lable_ILD_test=[]
features_ILD_test=[]



lables_name=os.listdir('features/Train_ILD')
try: lables_name.remove('.DS_Store')
except: pass
lables_dict={}
for i in range(len(lables_name)):
    lables_dict[lables_name[i]]=i+1

for ll in lables_name:
    lable_path=os.path.join('features/Train_ILD',ll)
    files=os.listdir(lable_path)
    try: files.remove('.DS_Store')
    except: pass
    
    tmp=[]
    for row in files:
        file_path=os.path.join(lable_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_ILD.append(feature_vector[0])
        tmp.append(feature_vector[0])
    
    np.random.shuffle(tmp)
    idx=len(tmp)
    features_ILD_train+=tmp[0:int(0.8*idx)]
    lable_ILD_train+=[lables_dict[ll]]*int(0.8*idx)
    
    features_ILD_test+=tmp[int(0.8*idx):]
    lable_ILD_test+=[lables_dict[ll]]*(idx-int(0.8*idx))

    lable_ILD+=[lables_dict[ll]]*len(files)

features_ILD=np.array(features_ILD)
lable_ILD=np.array(lable_ILD)

lable_ILD_test=np.array(lable_ILD_test)
lable_ILD_train=np.array(lable_ILD_train)

features_ILD_train=np.array(features_ILD_train)
features_ILD_test=np.array(features_ILD_test)


######## Talisman ##########
lable_Talisman=[]
features_Talisman=[]

lable_Talisman_train=[]
features_Talisman_train=[]

lable_Talisman_test=[]
features_Talisman_test=[]


for ll in lables_name:
    lable_path=os.path.join('features/Test_Talisman',ll)
    files=os.listdir(lable_path)
    try: files.remove('.DS_Store')
    except: pass

    tmp=[]
    for row in files:
        file_path=os.path.join(lable_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_Talisman.append(feature_vector[0])
        tmp.append(feature_vector[0])
    
    np.random.shuffle(tmp)
    idx=len(tmp)
    features_Talisman_train+=tmp[0:int(0.8*idx)]
    lable_Talisman_train+=[lables_dict[ll]]*int(0.8*idx)
    
    features_Talisman_test+=tmp[int(0.8*idx):]
    lable_Talisman_test+=[lables_dict[ll]]*(idx-int(0.8*idx))

    lable_Talisman+=[lables_dict[ll]]*len(files)

features_Talisman=np.array(features_Talisman)
lable_Talisman=np.array(lable_Talisman)

lable_Talisman_test=np.array(lable_Talisman_test)
lable_Talisman_train=np.array(lable_Talisman_train)

features_Talisman_test=np.array(features_Talisman_test)
features_Talisman_train=np.array(features_Talisman_train)




# clf = svm.SVC(decision_function_shape='ovo'  , probability=True,class_weight='balanced' ,verbose=True ,max_iter=-1, random_state=random.randint(1,900))
clf = svm.SVC(decision_function_shape='ovo'  , class_weight='balanced'  ,max_iter=-1)
clf.fit(features_ILD_train, lable_ILD_train)



####### ACC on Test data ########
label_predict_Test=[]
for row in features_ILD_test:
    ll=clf.predict([row])
    label_predict_Test.append(ll)


label_predict_Test=np.array(label_predict_Test)
label_predict_Test=label_predict_Test.reshape(lable_ILD_test.shape)
diff=(label_predict_Test==lable_ILD_test)
acc=100*np.sum(diff)/len(lable_ILD_test) 

print('Accuracy in ILD Test: ',acc)






# clf = svm.SVC(decision_function_shape='ovo'  , probability=True,class_weight='balanced' ,verbose=True ,max_iter=-1, random_state=random.randint(1,900))
clf = svm.SVC(decision_function_shape='ovo'  , class_weight='balanced'  ,max_iter=-1)
clf.fit(features_Talisman_train, lable_Talisman_train)



####### ACC on Test data ########
label_predict_Test=[]
for row in features_Talisman_test:
    ll=clf.predict([row])
    label_predict_Test.append(ll)


label_predict_Test=np.array(label_predict_Test)
label_predict_Test=label_predict_Test.reshape(lable_Talisman_test.shape)
diff=(label_predict_Test==lable_Talisman_test)
acc=100*np.sum(diff)/len(lable_Talisman_test) 

print('Accuracy in Talisman Test: ',acc)
