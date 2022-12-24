import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import svm
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import marketing_svm as svm_test
import marketing_fl as fl_test

#data preprocess
print("若符合要求之選項請輸入 1, 若不符合請輸入 0")
name = input('請輸入商品名稱：')
eat = input('是否為「吃的」：')
ingred = input('是否為「食材類」：')
drink = input('是否為「喝的」：')
cold = input('是否為「冰的」：')
hot = input('是否為「熱的」：')
sweet = input('是否為「甜的」：')
salty = input('是否為「鹹的」：')
fast = input('是否為「可立即食用」：')
daily = input('是否為「日常用品」：')
storage = input('是否為「收納類」：')
clear = input('是否為「清潔類」：')
learing = input('是否為「學習類」：')
eletrical = input('是否為「電器類」：')
bath = input('是否為「洗澡用品」：')
toil = input('是否為「個人衛生用品」：')
kitchen = input('是否為「廚房用品」：')
price_i = int(input('價錢為：'))


if price_i<30:
    price_i = 0
elif 30<=price_i<60:
    price_i = 1
elif 60<=price_i<90:
    price_i = 2  
elif 90<=price_i :
    price_i = 3

new_item = np.array([[eat,ingred,drink,cold,hot,sweet,salty,fast,daily,storage,clear,learing,eletrical,bath,toil,kitchen,price_i]],dtype=np.int8)

data_vec = pd.read_csv("market_data_vector拷貝.csv", sep=',') #把csv檔讀成dataframe

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
X_train, X_validation, y_train, y_validation = train_test_split(X, q, test_size = 0.33, random_state = 42)

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


#開始訓練

batch_size = 3213
epochs = 50

model1 = Sequential() 
model1.add(Dense(16, input_dim=17, activation='relu'))  
model1.add(Dense(5, activation='relu'))
model1.add(Dense(6, activation='relu'))
model1.add(Dense(5, activation='relu'))
model1.add(Dense(2,activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])  

history = model1.fit(X_train, y_train, batch_size, epochs,validation_data = (X_validation, y_validation)) 


fl_test.FL(data_vec,new_item)
svm_test.svm_function(data_vec,new_item)

print("================================================")
eval = model1.evaluate(X_train,y_train,verbose=0)
print("DNN eval in train data: loss = %0.6f, accuracy = %0.2f%%" %(eval[0],eval[1]*100))

eval = model1.evaluate(X_validation,y_validation,verbose=0)
print("DNN eval in test data: loss = %0.6f, accuracy = %0.2f%%" %(eval[0],eval[1]*100))

eval = model1.evaluate(food_val,food_ans,verbose=0)
print("DNN eval in food data: loss = %0.6f, accuracy = %0.2f%%" %(eval[0],eval[1]*100))

eval = model1.evaluate(daily_val,daily_ans,verbose=0)
print("DNN eval in daily data: loss = %0.6f, accuracy = %0.2f%%" %(eval[0],eval[1]*100))



pred = model1.predict(new_item)

if float(pred[0][0]) < float(pred[0][1]):
    print("DNN模型推測此商品較適合在此商店販售")
else:
    print("DNN模型推測此商品較不適合在此商店販售")
print("================================================")

print("================================================")
item_cat = ['吃的','食材','喝的','冰的','熱的','甜的','鹹的','可即食的','日常用品','收納類','清潔類','學習類','電器類','洗澡用品','衛生用品','廚房用品','貴的']

print(name, "的特徵為：",end="")
for i in range(17):
    if new_item[0][i] >= 1:
        print(item_cat[i],end=' ')
print("")
print("類似的商品為：")
item_cnt = 0
ratio_sum = 0

for i in range(4796):
    flag = 0
    for j in range(17):
        if new_item[0][j] != data_vec.iloc[i,3+j]:
            flag = 1
    if flag == 0:
        item_cnt = item_cnt + 1
        ratio_sum = ratio_sum + data_vec.iloc[i][20]
        print(data_vec.iloc[i][1]," ",data_vec.iloc[i][20])

print("")
print("================================================")
if item_cnt == 0:
    print("No same vector item")
else:
    avg = ratio_sum / item_cnt
    print("avg of the same vector item:",avg)
print("================================================")
