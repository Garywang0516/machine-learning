import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se 
import sklearn.metrics as sm

boston = sd.load_boston()
# 打亂序列
x,y = su.shuffle(boston.data, boston.target, random_state=7)
train_size = int(len(x) * 0.8)
# 前80％做訓練,後20％做測試
train_x,test_x,train_y,test_y = x[:train_size],x[train_size:],y[:train_size],y[train_size:]
# 決策樹回歸器
model = st.DecisionTreeRegressor(max_depth = 4)
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))
# 基於決策樹的正向激勵回歸器
model = se.AdaBoostRegressor(
	st.DecisionTreeRegressor(max_depth = 4),
	n_estimators = 400,
	random_state=7
	)
model.fit(train_x,train_y)
pred_test_y = model.predict(test_x)
print(sm.r2_score(test_y,pred_test_y))