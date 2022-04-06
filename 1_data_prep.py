


import os
import shutil

shutil.rmtree('data')
if not os.path.isdir('data'): os.mkdir('data')
if not os.path.isdir('data/Train_ILD'): os.mkdir('data/Train_ILD')
if not os.path.isdir('data/Test_Talisman'): os.mkdir('data/Test_Talisman')



os.system('python3 data_prep/gen_patch.py')
os.system('python3 data_prep/preprocess_Talisman.py')



src=[
    'data_prep/OUTPUT_PATCH/patches_norm/micronodules',
    'data_prep/OUTPUT_PATCH/patches_norm/healthy',
    'data_prep/OUTPUT_PATCH/patches_norm/fibrosis',
    'data_prep/OUTPUT_PATCH/patches_norm/emphysema',
    'data_prep/OUTPUT_PATCH/patches_norm/ground_glass']

dst=[
    'data/Train_ILD/micronodules',
    'data/Train_ILD/healthy',
    'data/Train_ILD/fibrosis',
    'data/Train_ILD/emphysema',
    'data/Train_ILD/ground']



for i,row in enumerate(src):
    shutil.move(row, dst[i])
shutil.move('data_prep/OUTPUT_PATCH/patches_norm/raw_stat.txt', 'data/raw_stat.txt')




############# statistics #############
def file_stat(pp):
    labels=[]
    for row in os.listdir(pp):
        if row == '.DS_Store':
            continue
        tmp=os.listdir(os.path.join(pp,row))
        labels.append([row,len(tmp)])

    labels.sort(key=lambda x:x[1],reverse=True)
    return labels

labels=file_stat('data/Train_ILD')
with open('data/Statistics_Train.txt', 'w') as f:
    for line in labels:
        f.write(line[0]+': '+str(line[1]))
        f.write('\n')

labels=file_stat('data/Test_Talisman')
with open('data/Statistics_Test.txt', 'w') as f:
    for line in labels:
        f.write(line[0]+': '+str(line[1]))
        f.write('\n')



try:
    shutil.rmtree('data_prep/OUTPUT_PATCH')
    shutil.rmtree('__pycache__')
    shutil.rmtree('data_prep/__pycache__')
except: pass
