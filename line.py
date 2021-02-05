import numpy as np 
import sklearn.linear_model as lm 
import sklearn.metrics as sm
import matplotlib.pyplot as mp 
x,y = [],[]
with open('/home/ubuntu/桌面/machine learning/data/single.txt','r') as f:
	for line in f.readlines():
		data = [float(substr) for substr in line.split(',')]
		x.append(data[:-1])
		y.append(data[-1])
		
x = np.array(x)
y = np.array(y)
model = lm.LinearRegression()
model.fit(x,y)
pred_y = model.predict(x)
print(sm.r2_score(y,pred_y))

mp.figure("linear regression",facecolor='lightgray')
mp.title('linear regression',fontsize = 20)
mp.xlabel('x',fontsize=14)
mp.ylabel('y',fontsize=14)
mp.grid(linestyle=":")
mp.scatter(x,y,c='dodgerblue',alpha=0.75,s=60,
	label='Sample')
# ravel -> 將數組變為一維  argsort -> 將x中的元素從小到大排列，提取其對應的index，然後輸出到y
sorted_indies = x.ravel().argsort()
mp.plot(x[sorted_indies],pred_y[sorted_indies],
	c='orangered',label='Regression')
mp.legend()
mp.show()