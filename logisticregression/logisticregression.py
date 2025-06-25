# # import ccxt
# # import pandas as pd
# # import time
# # import matplotlib.pyplot as plt

# # kucoin = ccxt.kucoin()

# # now = int(time.time()* 1000)

# # start_time = now - (1000 * 60 * 60 * 1000)

# # ohlcv = kucoin.fetch_ohlcv('DOGE/USDT', timeframe='1h', since=start_time, limit=1000)

# # df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
# # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# # df.to_csv('doge.csv', index=False)
# # print('data saved')

# # df = pd.read_csv('doge.csv')
# # df['buy_class'] = (df['close']>df['open']).astype(int)

# # buy_df = df[df['buy_class'] == 1]
# # plt.figure(figsize=(12, 6))
# # plt.scatter(buy_df['timestamp'], buy_df['volume'], color='green', label='Buy Class', alpha=0.6)
# # plt.xlabel('Timestamp')
# # plt.ylabel('Volume')
# # plt.title('Buy Class - Volume vs Time')
# # plt.xticks(rotation=45)
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# # ---------------------------------------using titanic dataset for logisticregression---------------------->

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# df = pd.read_csv('titanic.csv')
# df.info()
# print(df.describe())--------------------->Gives summary statistics for all numeric columns:
# print(df.isnull().sum())----------------->Tells you how many missing (NaN) values are in each column.

# droping unecessary colums atleast they are not good right now

# df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# ''' this is telling remove this list of names and axis is telling these are columns by giving
#     num 1, and 0 is for rows and inplace is for directing changes to the dataframe permenetly '''

# #  filling missing value means ages with mean
# df['Age'].fillna(df['Age'].mean(), inplace=True)

# #  filling embark missing value to most common values
# df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# # common sex to numaric 
# df['Sex'] = df['Sex'].map({'male':0, 'female':1})
# print(df.head())

# # convert embarked to numaric
# df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2})

# df.to_csv('titanic_clean.csv', index=False)

# print(df.isnull().sum())
# print(df.describe())

df = pd.read_csv('titanic_clean.csv')
x = df[['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print(f"accuracy:{accuracy:.2f}")

y_pred = model.predict(x_test)
# print(y_pred)
cm = confusion_matrix(y_test,y_pred)
print(cm)

# plt.figure(figsize=(5,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0 (Died)', 'Predicted 1 (Survived)'],
#             yticklabels=['Actual 0 (Died)', 'Actual 1 (Survived)'])
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.savefig('confusion_matrix.png')
# plt.show()

for feature, weight in zip(x.columns, model.coef_[0]):
    print(f"{feature}: {weight:.4f}")

z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid)
plt.title("Sigmoid Curve")
plt.xlabel("z (Linear Output)")
plt.ylabel("Sigmoid(z) = Probability")
plt.grid(True)
plt.savefig('sigmoid_function.png')
plt.show()