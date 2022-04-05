

import os
import scipy.io
import numpy as np

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



