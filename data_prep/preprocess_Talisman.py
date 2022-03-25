import os
import cv2
import numpy as np

path='ILD_DB/ILD_DB_talismanTestSuite'
new_path='data/Test_Talisman'

if not os.path.isdir(new_path): os.mkdir(new_path)  
for row in os.listdir(path):
    [cat,name_old]=row.split('_',1)
    name_new=name_old.replace('tif','png')
    cur_path=os.path.join(new_path,cat)
    if not os.path.isdir(cur_path): os.mkdir(cur_path)  

    img=cv2.imread(os.path.join(path,row),-1)
    img=np.float64(img)
    img=img-np.amin(img)
    img=img/np.amax(img)
    img=255*img
    img=np.uint8(img)
    cv2.imwrite(os.path.join(cur_path,name_new),img)




