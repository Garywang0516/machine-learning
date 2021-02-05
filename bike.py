import sklearn.utils as su
import sklearn.ensemble as se 
import matplotlib.pyplot as mp 
import numpy as np
import csv
import sklearn.metrics as sm
with open('/home/ubuntu/桌面/machine learning/data/bike_day.csv','r') as f:
	reader = csv.reader(f)
	x,y, =[],[]
	for row in reader:
		x.append(row[2:13])
		y.append(row[-1])
fn_dy = np.array(x[0])
x = np.array(x[1:],dtype = float)
y = np.array(y[1:],dtype = float)
x,y = su.shuffle(x,y,random_state = 7)


train_size = int(len(x) * 0.9)
# 前90％做訓練,後10％做測試
train_x,test_x,train_y,test_y = x[:train_size],x[train_size:],y[:train_size],y[train_size:]
# 隨機森林回歸器
model = se.RandomForestRegressor(
	max_depth = 10,
	n_estimators = 1000,
	min_samples_split = 2
	)

model.fit(train_x,train_y)
# 基於"天"的數據集的特徵重要性
fi_dy = model.feature_importances_
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))

with open('/home/ubuntu/桌面/machine learning/data/bike_hour.csv','r') as f:
	reader = csv.reader(f)
	x,y, =[],[]
	for row in reader:
		x.append(row[2:13])
		y.append(row[-1])
fn_hr = np.array(x[0])
x = np.array(x[1:],dtype = float)
y = np.array(y[1:],dtype = float)
x,y = su.shuffle(x,y,random_state = 7)


train_size = int(len(x) * 0.9)
# 前90％做訓練,後10％做測試
train_x,test_x,train_y,test_y = x[:train_size],x[train_size:],y[:train_size],y[train_size:]
# 隨機森林回歸器
model = se.RandomForestRegressor(
	max_depth = 10,
	n_estimators = 1000,
	min_samples_split = 2
	)

model.fit(train_x,train_y)
# 基於"小時"的數據集的特徵重要性
fi_hr = model.feature_importances_
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))

mp.figure('Bike',facecolor ='lightgray')
mp.subplot(211)
mp.title('Day',fontsize=16)
mp.ylabel('Importance',fontsize=16)
mp.tick_params(labelsize=10)
mp.grid(axis='y',linestyle=':')
sorted_indices = fi_dy.argsort()[::-1]
pos = np.arange(sorted_indices.size)
mp.bar(pos,fi_dy[sorted_indices],facecolor='deepskyblue',edgecolor='steelblue')
mp.xticks(pos,fn_dy[sorted_indices],rotation=30)

mp.subplot(212)
mp.title('Hour',fontsize=16)
mp.ylabel('Importance',fontsize=16)
mp.tick_params(labelsize=10)
mp.grid(axis='y',linestyle=':')
sorted_indices = fi_hr.argsort()[::-1]
pos = np.arange(sorted_indices.size)
mp.bar(pos,fi_hr[sorted_indices],facecolor='lightcoral',edgecolor='indianred')
mp.xticks(pos,fn_hr[sorted_indices],rotation=30)
mp.tight_layout()
mp.show()