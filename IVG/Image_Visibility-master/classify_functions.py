import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


labels_name=['healthy', 'ground', 'micronodules', 'emphysema', 'fibrosis']
labels_dict={'healthy': 1, 'ground': 2, 'micronodules': 3, 'emphysema': 4, 'fibrosis': 5}


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


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
            number=100*CM[i, j].copy()
            if number==float(0):
                number=int(number)
            number=round(number,2)
            if(CM[i, j] > thresh) : color="white"
            else: color="black"
            fontweight='normal'
            fontsize=20
            if float(number)>50 : 
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

