

import os
import scipy.io
import numpy as np
from sklearn import svm
import random


######## Train ##########
lables=[]
features=[]

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
    
    for row in files:
        file_path=os.path.join(lable_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features.append(feature_vector[0])

    lables+=[lables_dict[ll]]*len(files)

features=np.array(features)
lables=np.array(lables)


######## Test ##########
lables_Test=[]
features_Test=[]
for ll in lables_name:
    lable_path=os.path.join('features/Test_Talisman',ll)
    files=os.listdir(lable_path)
    try: files.remove('.DS_Store')
    except: pass
    
    for row in files:
        file_path=os.path.join(lable_path,row)
        feature_vector = scipy.io.loadmat(file_path)['feature_vector']
        features_Test.append(feature_vector[0])

    lables_Test+=[lables_dict[ll]]*len(files)

features_Test=np.array(features_Test)
lables_Test=np.array(lables_Test)






# clf = svm.SVC(decision_function_shape='ovo'  , probability=True,class_weight='balanced' ,verbose=True ,max_iter=-1, random_state=random.randint(1,900))
clf = svm.SVC(decision_function_shape='ovo'  , class_weight='balanced'  ,max_iter=-1)
clf.fit(features, lables)



####### ACC on Test data ########
label_predict_Test=[]
for row in features_Test:
    ll=clf.predict([row])
    label_predict_Test.append(ll)


label_predict_Test=np.array(label_predict_Test)
label_predict_Test=label_predict_Test.reshape(lables_Test.shape)
diff=(label_predict_Test==lables_Test)
acc=100*np.sum(diff)/len(lables_Test) 

print('Accuracy in Test: ',acc)
