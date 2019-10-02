import numpy as np
from tensorflow.keras.datasets import mnist
from logisticRegressor import logisticRegressor

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, 28*28))/255
x_test = x_test.reshape((10000, 784))/255

print(f"Xtrain shape: {x_train.shape}")
print(f"Ytrain shape: {y_train.shape}")

classes, features, reg = 10, 784, 0.1
model = logisticRegressor(classes, features, reg)
data = (x_train, y_train), (x_test, y_test)

history = model.fit(data, val=True, learning_rate=0.15, epochs=200)

print("------------------------------")
print("Train accuracy:", model.accuracy(x_train, y_train))
print("Validation acc:", model.accuracy(x_test, y_test))


m = 10
indices, proba = model.predict(x_test[:m], prob=True)

print("------------------------------")
print("Actual\tPredicted Confidence")
for i in range(m):
    print(f"{y_train[i]:02}\t{indices[i]:02}\t  {proba[i]}")

print("-------------------------------")

model.plot_history(history)