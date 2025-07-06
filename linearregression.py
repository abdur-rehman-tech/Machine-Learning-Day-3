from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

#sample Data
X=np.array([[1400],[1600],[1700],[1875],[1100],[1550],[2350],[2450]])
Y=np.array([245000,312000,279000,308000,199000,219000,405000,324000])

#spliting Data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#intilize model
model=LinearRegression()
model.fit(X_train,Y_train)

#prediction
Y_pred=model.predict(X_test)

#evaluate
mse=mean_squared_error(Y_test,Y_pred)
print("Mean squared: ",mse)
print("Predicted Value: ",Y_pred)

plt.scatter(X_test, Y_test, color='black')
plt.plot(X_test, Y_pred, color='blue', linewidth=3)
plt.title("Linear Regression")
plt.xlabel("Input")
plt.ylabel("Target")
plt.show()