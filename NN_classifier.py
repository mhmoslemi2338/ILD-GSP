






[ILD_train,ILD_test]=make_train_test_Data(features_ILD,label_ILD,48,0.9)

train=ILD_train.copy()
test=ILD_test.copy()

x_train =  train.loc[:,[i for i in range(train.shape[1]-1)]].values
y_train = train.loc[:,[train.shape[1]-1]].values.ravel()

x_test=  test.loc[:,[i for i in range(test.shape[1]-1)]].values
y_test = test.loc[:,[test.shape[1]-1]].values.ravel()



[ILD_train,ILD_validation]=make_train_test_Data(x_train,y_train.reshape(-1,1),48,0.7)



train=ILD_train.copy()
test=ILD_validation.copy()

x_train =  train.loc[:,[i for i in range(train.shape[1]-1)]].values
y_train = train.loc[:,[train.shape[1]-1]].values.ravel()

x_val=  test.loc[:,[i for i in range(test.shape[1]-1)]].values
y_val = test.loc[:,[test.shape[1]-1]].values.ravel()





epochs_num=20

def my_convert(arr_in):
    tmp=np.zeros((arr_in.shape[0],5))
    for i,row in enumerate(arr_in):
        tmp[i][int(row)-1]=1 
    return tmp.copy()
    

from tensorflow.keras import models, layers ,Sequential ,losses , optimizers
import tensorflow as tf

with tf.device('/GPU:0'):


    ### layer input
    inputs = layers.Input(x_train.shape[1:])
    ### hidden layers
    hidden1 = layers.Dense(10,name="hidden1",activation='sigmoid')(inputs)
    hidden2 = layers.Dense(10,name="hidden2", activation='sigmoid')(hidden1)
    # hidden3 = layers.Dense(30,name="hidden3", activation='sigmoid')(hidden2)
    ### layer output
    outputs = layers.Dense(5,name="output", activation='softmax')(hidden2)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


    history = model.fit(x_train, my_convert(y_train), validation_data=(x_val,my_convert(y_val)),epochs=epochs_num, verbose=1)




with tf.device('/GPU:0'):
    Test_loss, Test_acc = model.evaluate(x_test, my_convert(y_test), verbose=1)
    print('\nTest loss:',str(round(Test_loss,5)))
    print('Test Accuracy:',str(round(100*Test_acc,4)),' %\n')

fig=plt.figure(figsize=(15,12))
plt.subplot(211)
plt.plot(range(1, epochs_num+1), np.array(history.history['accuracy'])*100, '*--', label='Training acc')
plt.plot(range(1, epochs_num+1), np.array(history.history['val_accuracy'])*100, '*--', label='Validation acc')

plt.title('Training and Validation Accuracy',fontsize='x-large'); 
plt.legend();plt.grid();plt.xlabel("Epoch",fontsize='x-large'); plt.ylabel("Acuuracy",fontsize='x-large')
plt.subplot(212)
plt.plot(range(1, epochs_num+1), history.history['loss'], '*--', label='Training loss')
plt.plot(range(1, epochs_num+1), history.history['val_loss'], '*--', label='Validation loss')
plt.title('Training and Validation loss',fontsize='x-large')
plt.grid();plt.xlabel("Epoch",fontsize='x-large'); plt.ylabel("Loss",fontsize='x-large'); plt.legend()
plt.show()
