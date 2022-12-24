
import numpy as np
import pandas as pd
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dense
import tensorflow.compat.v2 as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

def FL(data_vec,new_item):

    y = data_vec[['ratio']].values.reshape(-1, 1)

    q = []
    for i in range(4796):
        if float(y[i]) > 0.02067:
            q.append([0,1])
        else:
            q.append([1,0])

    q = np.array(q)


    val = data_vec[['貴']].values
    price = data_vec[['$$']].values

    for i in range(4796):
        if int(price[i])<30:
            val[i] = 0
        elif 30<=int(price[i])<60:
            val[i] = 1
        elif 60<=int(price[i])<90:
            val[i] = 2
        elif 90<=int(price[i]) :
            val[i] = 3
    for i in range(4796):
        data_vec.loc[i,'貴'] = val[i]

    X = data_vec.drop(['Num','名稱','$$','ratio'], axis = 1).values

    X_train, X_validation, y_train, y_validation = train_test_split(X, q, test_size = 0.1, random_state = 42)

    daily_com_val = []  #日常用品驗證集
    food_com_val = []   #食品驗證集
    daily_com_ans = []  
    food_com_ans = []

    cnt = 0

    for i in X_validation:
        if i[8] == 1:
            daily_com_val.append(i)
            daily_com_ans.append(y_validation[cnt])
        else:
            food_com_val.append(i)
            food_com_ans.append(y_validation[cnt])
        cnt = cnt + 1 

    daily_val = np.array(daily_com_val)
    daily_ans = np.array(daily_com_ans)
    food_val = np.array(food_com_val)
    food_ans = np.array(food_com_ans)

    #client1   #分四半
    x_t1 = X_train[0:1079] 
    y_t1 = y_train[0:1079]

    x_t2 = X_train[1079:2158] 
    y_t2 = y_train[1079:2158]

    x_t3 = X_train[2158:3237] 
    y_t3 = y_train[2158:3237]

    x_t4 = X_train[3237:4316] 
    y_t4 = y_train[3237:4316]

    #client1
    model1 = Sequential()
    model1.add(Dense(100, input_dim=17, activation='relu'))
    model1.add(Dense(100, activation='relu'))
    model1.add(Dense(10, activation='relu')) 
    model1.add(Dense(2,activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])  

    model1.fit(x_t1, y_t1, batch_size=100, epochs=20) 

    #client2
    model2 = Sequential()
    model2.add(Dense(100, input_dim=17, activation='relu'))
    model2.add(Dense(100, activation='relu'))
    model2.add(Dense(10, activation='relu')) 
    model2.add(Dense(2,activation='softmax'))
    model2.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])  

    model2.fit(x_t2, y_t2, batch_size=100, epochs=20) 


    #client3
    model3 = Sequential()
    model3.add(Dense(100, input_dim=17, activation='relu'))
    model3.add(Dense(100, activation='relu'))
    model3.add(Dense(10, activation='relu')) 
    model3.add(Dense(2,activation='softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])  

    model3.fit(x_t3, y_t3, batch_size=100, epochs=20) 


    #client4
    model4 = Sequential() #打開空白神經網路學習機 
    model4.add(Dense(100, input_dim=17, activation='relu')) #100 是指神經元數量 ,input_dim是輸入大小 #relu: 神經網路激發函數
    model4.add(Dense(100, activation='relu'))
    model4.add(Dense(10, activation='relu')) 
    model4.add(Dense(2,activation='softmax'))
    model4.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    model4.fit(x_t4, y_t4, batch_size=100, epochs=20) 

    w = [0]*8
    for i in range(8):
        w[i] = model1.get_weights()[i]+model2.get_weights()[i]+model3.get_weights()[i]+model4.get_weights()[i]
        w[i] = w[i]/4.0

    #client1 retraining #time counting
    model1.set_weights(w)

    print("================================================")
    eval = model1.evaluate(X_train,y_train,verbose=0)
    print("FL eval in train data: loss = %0.6f, accuracy = %0.2f%%" %(eval[0],eval[1]*100))
    eval = model1.evaluate(X_validation,y_validation,verbose=0)
    print("FL eval in test data: loss = %0.6f, accuracy = %0.2f%%" %(eval[0],eval[1]*100))
    eval = model1.evaluate(food_val,food_ans,verbose=0)
    print("FL eval in food data: loss = %0.6f, accuracy = %0.2f%%" %(eval[0],eval[1]*100))
    eval = model1.evaluate(daily_val,daily_ans,verbose=0)
    print("FL eval in daily data: loss = %0.6f, accuracy = %0.2f%%" %(eval[0],eval[1]*100))

    pred = model1.predict(new_item)

    if float(pred[0][0]) < float(pred[0][1]):
        print("FL模型推測此商品較適合在此商店販售")
    else:
        print("FL模型推測此商品較不適合在此商店販售")

    print("================================================")

