import pandas as pd
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\mine\UESTC\S2.part2\NTU\1\DBS_SingDollar.csv")
# print(df)
X = df.loc[:, ["SGD"]] # 每一行的 SGD 列
Y = df.loc[:, ["DBS"]]
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, Y)
# print(model.coef_)
# print(model.intercept_)
pred = model.predict(X)
# print(pred)
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(Y, pred) ** 0.5
# print(rmse)

from sklearn import tree
import joblib
joblib.dump(model, 'regression.jl')
model = tree.DecisionTreeRegressor()
model.fit(X, Y)
pred = model.predict(X)
rmse = mean_squared_error(Y, pred) ** 0.5
print(rmse)
joblib.dump(model, "decisiontree.jl")