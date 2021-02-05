# # 梯度下降法
# # 在機器學習算法中，對於很多監督學習模型，需要對原始的模型構建損失函數，
# 接下來便是通過優化算法對損失函數進行優化，以便尋找到最優的參數。
# 在求解機器學習參數的優化算法中，使用較多的是基於梯度下降的優化算法(Gradient Descent, GD)。

# # 梯度下降法有很多優點，其中，在梯度下降法的求解過程中，只需求解損失函數的一階導數，
# 計算的代價比較小，這使得梯度下降法能在很多大規模數據集上得到應用。
# 梯度下降法的含義是通過當前點的梯度方向尋找到新的疊代點。
# # 基本思想可以這樣理解：我們從山上的某一點出發，找一個最陡的坡走一步（也就是找梯度方向），
# 到達一個點之後，再找最陡的坡，再走一步，直到我們不斷的這麼走，走到最「低」點（最小花費函數收斂點）。
import numpy as np 
import matplotlib.pyplot as mp 
from mpl_toolkits.mplot3d import axes3d
train_x = np.array([0.5,0.6,0.8,1.1,1.4])
train_y = np.array([5.0,5.5,6.0,6.8,7.0])
n_epoches = 1000
lrate = 0.01
epoches,losses = [],[]
w0,w1 = [1],[1]
for epoch in range(1,n_epoches+1):
	epoches.append(epoch)
	losses.append(((train_y - (w0[-1] + w1[-1] * train_x)) ** 2/2).sum())
	print('{:4}> w0={:.8f},w1={:.8f},loss={:.8f}'.format(
		epoches[-1], w0[-1], w1[-1], losses[-1]))

	d0 = -(train_y - (
		w0[-1] + w1[-1] * train_x)).sum()
	d1 = -((train_y - (
			w0[-1] + w1[-1] * train_x))* train_x).sum()
	w0.append(w0[-1]-lrate * d0)
	w1.append(w1[-1]-lrate * d1)
w0 = np.array(w0[:-1])
w1 = np.array(w1[:-1])
sorted_indices = train_x.argsort()
test_x = train_x[sorted_indices]
test_y = train_y[sorted_indices]
pred_test_y = w0[-1]+ w1[-1] * test_x

grid_w0,grid_w1 = np.meshgrid(
	np.linspace(0,9,500),
	np.linspace(0,3.5,500))
flat_w0 ,flat_w1 = grid_w0.ravel(),grid_w1.ravel()
flat_loss = (((flat_w0 + np.outer(
	train_x,flat_w1)) - train_y.reshape(-1,1)) ** 2).sum(axis = 0) / 2
grid_loss = flat_loss.reshape(grid_w0.shape)

mp.figure('Linear Regression',facecolor = 'lightgray')
mp.title('Linear Regression',fontsize=20)
mp.xlabel('x',fontsize=14)
mp.ylabel('y',fontsize=14)
mp.grid(linestyle=':')
mp.scatter(train_x,train_y,marker='s',
	c='dodgerblue',alpha=0.5,s=80,
	label='Training')
mp.scatter(test_x,test_y,marker='d',
	c='orangered',alpha=0.5,s=60,
	label='Testing')
mp.scatter(test_x,pred_test_y,
	c='orangered',alpha=0.5,s=60,
	label='Testing')
for x,y,pred_y in zip(test_x,test_y,pred_test_y):
	mp.plot([x,x],[y,pred_y],c='orangered',
		alpha=0.5,linewidth=1)
mp.plot(test_x,pred_test_y,'--',c = 'limegreen',
	label='Regression',linewidth=1)
mp.legend()

mp.figure('Training progress',facecolor = 'lightgray')
mp.subplot(311)
mp.title('Training progress',fontsize=20)
mp.ylabel('w0',fontsize=14)
mp.gca().xaxis.set_major_locator(
	mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches,w0,c = 'dodgerblue',
	label='w0',linewidth=1)
mp.tight_layout()
mp.legend()

mp.subplot(312)
mp.ylabel('w0',fontsize=14)
mp.gca().xaxis.set_major_locator(
	mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches,w1,c = 'limegreen',
	label='w1',linewidth=1)
mp.tight_layout()
mp.legend()

mp.subplot(313)
mp.xlabel('epoch',fontsize=14)
mp.ylabel('loss',fontsize=14)
mp.gca().xaxis.set_major_locator(
	mp.MultipleLocator(100))
mp.grid(linestyle=':')
mp.plot(epoches,losses,c = 'orangered',
	label='losses',linewidth=1)
mp.tight_layout()
mp.legend()

mp.figure('Loss Function')
ax = mp.gca(projection = '3d')
mp.title('Loss Function',fontsize = 20)
ax.set_xlabel('w0',fontsize = 14)
ax.set_ylabel('w1',fontsize = 14)
ax.set_zlabel('loss',fontsize = 14)
mp.tick_params(labelsize = 10)
ax.plot_surface(grid_w0,grid_w1,grid_loss,
	rstride = 10,cstride = 10,cmap = 'jet')
ax.plot(w0,w1,losses,'o-',c= 'orangered',
	label='BGD')
mp.legend()

mp.figure('Batch Gradnient Descent',facecolor = 'lightgray')
mp.title('Batch Gradnient Descent',fontsize=20)
mp.xlabel('x',fontsize=14)
mp.ylabel('y',fontsize=14)
mp.tick_params(labelsize = 10)
mp.grid(linestyle=':')
mp.contour(grid_w0,grid_w1,grid_loss,1000,cmap = 'jet')
cntr = mp.contour(grid_w0,grid_w1,grid_loss,10,colors = 'black',linewidths=0.5)
mp.clabel(cntr,inline_spacing = 0.1,fmt ='%.2f',
	fontsize = 8)
mp.plot(w0,w1,'o-',c = 'orangered',label = 'BDG')
mp.legend()
mp.show()