
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import  confusion_matrix
from sklearn import svm
from classify_functions import *

try: os.mkdir('results')
except: pass


########### select Train and Test sets #############
# split data between train and test 
# we choose 25% of each class data for Test
# after selecting Train , test we shuffles each set using unison_shuffled_copies
####################################################
Data=pd.read_pickle('Data.pkl')
Data=Data.rename(columns={"label": 2112})
Train=Data.groupby(Data.shape[1]-1).apply(lambda s: s.sample(frac=0.75,replace=False,random_state=0))
Train = Train.reset_index(level=[None])
Train=Train.set_index('level_1')
Test= Data.drop(index=Train.index)

x_train =  Train.loc[:,[i for i in range(Train.shape[1]-1)]].values
y_train = Train.loc[:,[Train.shape[1]-1]].values.ravel()
x_train, y_train=unison_shuffled_copies(x_train, y_train)

x_test =  Test.loc[:,[i for i in range(Test.shape[1]-1)]].values
y_test = Test.loc[:,[Test.shape[1]-1]].values.ravel()
x_test, y_test=unison_shuffled_copies(x_test, y_test)

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
clf = svm.SVC(kernel='rbf',decision_function_shape='ovo'   ,max_iter=-1)
clf.fit(x_train, y_train)

#**** predict label for Test data
x_test=scaler.transform(x_test)
x_test=pca.transform(x_test)
label_predict_test=clf.predict(x_test)

#*****
CM=confusion_matrix(y_test, label_predict_test)
TP = np.diag(CM)
FP = np.sum(CM, axis=0) - TP
FN = np.sum(CM, axis=1) - TP
#*****
accuracy=100*np.sum(label_predict_test==y_test)/len(y_test)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1_score=100*np.average(2*precision*recall/(recall+precision))
#*****
print('Feature vector length:',Train.shape[1])
print('\nSVM, RBF kernel,',pca_num,'PCA components: ')
print('\tAccuracy on Test: ' ,round(accuracy,3),'%')
print('\tF1_score on Test: ',round(F1_score,3),'%')
#*****
plot_CM(CM,labels_name,'Confusion Matrix for ILD Test Data',True)
stat_ILD=statistics_CM(CM,labels_name,labels_dict)
with open('results/ILD_Test.txt', mode='w') as file_object:
    print(stat_ILD, file=file_object)
#*****