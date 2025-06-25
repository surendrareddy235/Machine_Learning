# import matplotlib.pyplot as plt
# height  = [150,160,165,170,175,180]
# weight = [50,60,63,65,70,75]

# x_mean = sum(height)/ len(height)
# y_mean = sum(weight)/ len(weight)

# numerator = sum((xi - x_mean) * (yi - y_mean) for xi,yi in zip(height,weight))
# denominator = sum((xi - x_mean) ** 2 for xi in height)
# m = numerator/denominator

# b_line = y_mean - m * x_mean
# print(f"Best-fit line: y = {m:.2f}x + {b_line:.2f}")
# predicted_weight = [m * h + b_line for h in height]
# print(m * 190 + b_line)


# plt.scatter(height,weight, color='red', label='data points')
# plt.plot(height,[m*xi + b_line for xi in height], label=f'y = {m:.2f}x +{b_line:.2f}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# ----------------------------linear regression using ml library------------->

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

height  = [150,160,165,170,175,180]
weight = [50,60,63,65,70,75]

x = np.array(height).reshape(-1,1)
y = np.array(weight)

model = LinearRegression()
model.fit(x,y)
print(f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")
print(model.predict([[180]]))

y_pred = model.predict(x)
print(y_pred)

mse = mean_squared_error(y,y_pred)
print(f"MSE={mse:.2f}")

r2 = r2_score(y,y_pred)
print(f"r_square={r2:.2f}")
# plt.scatter(x, y) 
# plt.plot(x, model.predict(x))
# plt.show()

