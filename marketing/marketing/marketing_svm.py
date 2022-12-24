import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


def svm_function(data_vec,new_item):
    # data_vec = pd.read_csv("market_data_vector拷貝.csv", sep=',') #把csv檔讀成dataframe

    y = data_vec[['ratio']].values.ravel()

    for i in range(4796):
        if float(y[i]) > 0.02067:
            y[i] = 1
        else:
            y[i] = 0

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

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.33, random_state = 42)

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


    clf=svm.SVC(kernel='rbf',C=1000,gamma=0.1) #(高斯函數>sigmoid) (c值：越大->容錯度越小)
    clf.fit(X_train,y_train)

    print("================================================")
    print("SVM eval in train data accuracy = %0.2f%%" %(clf.score(X_train, y_train)*100))

    print("SVM eval in validation data accuracy = %0.2f%%" %(clf.score(X_validation, y_validation)*100))

    print("SVM eval in food data accuracy = %0.2f%%" %(clf.score(food_val, food_ans)*100))

    print("SVM eval in daily data accuracy = %0.2f%%" %(clf.score(daily_val, daily_ans)*100))

    pred = clf.predict(new_item)


    if pred[0] == 1 :
        print("SVM模型推測此商品較適合在此商店販售")
    else:
        print("SVM模型推測此商品較不適合在此商店販售")
    print("================================================")