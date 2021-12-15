import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
# from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import datetime

# Load Dataset
data = pd.read_csv('dataset/005930.KS_5y.csv')
data.head()

# Compute Mid Price
high_prices = data['High'].values
low_prices = data['Low'].values
mid_prices = (high_prices + low_prices) / 2

# Create Windows
seq_len = 50
sequence_length = seq_len + 1

result = []
for index in range(len(mid_prices) - sequence_length):
    # mid 값을 한칸씩 밀려넣어둔다 [[51개씩],[]]
    result.append(mid_prices[index: index + sequence_length])

# Normalize Data
normalized_data = []
for window in result:
    normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# split train and test data
row = int(round(result.shape[0] * 0.9))  # 9 : 1 로 train 과 test 를 나눈다
train = result[:row, :]
np.random.shuffle(train)  # 트레인용데이타는 섞어준다.

x_train = train[:, :-1]  # [[]]  모든 로, 각로의 마지막 1은 제외(결과값 즉 y train 값)
# print('x_train.shape 1', x_train.shape) # x_train.shape (1057, 50)  2차원 배열 [1057][50]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]  # 모든로에서 마지막 하나(결과값)

x_test = result[row:, :-1]
# print('x_test.shape 1', x_test.shape)  # x_test.shape (117, 50)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

# print('x_train.shape 2', x_train.shape)  # x_train.shape (1057, 50, 1) -> 2차원배열이 3차원 배열로 변경 (LSTM 은 3차원 배열을 이용)
# print('x_test.shape 2', x_test.shape)  # x_test.shape (117, 50, 1)

# Build a Model
model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))  # unit 50개를 추론해 1개의 데이타를 추출한다는 의미
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

# Training
model.fit(x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=10,
    epochs=20)

# Prediction
# 이결과값은 Low High가 들어갔을때 실제 값을 마추는 것 같은데
pred = model.predict(x_test) ## 테스트 데이타

fig = plt.figure(facecolor='white', figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()
