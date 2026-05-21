import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import RandomNormal  # Import the initializer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from time import perf_counter


def getFormatedTime():
    return datetime.now().strftime("%Y%m%d%H%M%S")


epochs = 200
do_not_plot = 10
global_random_state = 1
tf.random.set_seed = global_random_state  # fix random state


df0 = pd.read_csv("train0.csv")
df1 = pd.read_csv("train1.csv")
df2 = pd.read_csv("train2.csv")
df = pd.concat([df0, df1, df2])
shuffled_df = df.sample(frac=1, random_state=1).reset_index(drop=True)
print(shuffled_df.head())

arr = df.to_numpy()
X = arr[:, 1:-1]
y = arr[:, -1].astype(np.int8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def train_on(
    L: list,
    u=do_not_plot,
    epochs=epochs,
    activation="relu",
    kernel_initializer=RandomNormal(seed=global_random_state),
):  # https://chatgpt.com/share/66fb9885-8ad8-8009-813c-e79678b29b19
    tin = perf_counter()
    model = Sequential(
        [
            Dense(units=c, activation=activation, kernel_initializer=kernel_initializer)
            for c in L[:-1]
        ]
        + [
            Dense(
                units=L[-1], activation="linear", kernel_initializer=kernel_initializer
            )
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )
    history = model.fit(
        X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0
    )

    def Min(u, L):
        return [min(u, c) for c in L]

    train_loss = Min(5, history.history["loss"][u:])
    test_loss = Min(5, history.history["val_loss"][u:])
    tout = perf_counter()
    print(
        f"train on {L} + {activation} + {kernel_initializer} successfull in {tout -tin}s"
    )

    return model, train_loss, test_loss


_, train_loss1_sig, test_loss1_sig = train_on([3, 5, 7, 5, 3], activation="sigmoid")
_, train_loss1_ge, test_loss1_ge = train_on(
    [3, 5, 7, 5, 3], activation="gelu"
)  # https://www.tensorflow.org/api_docs/python/tf/keras/activations/gelu
_, train_loss1_tanh, test_loss1_tanh = train_on([3, 5, 7, 5, 3], activation="tanh")

_, train_loss1, test_loss1 = train_on([3, 5, 7, 5, 3])
_, train_loss2, test_loss2 = train_on([3, 5, 3])
_, train_loss3, test_loss3 = train_on([50, 25, 12, 25, 50, 3])
_, train_loss4, test_loss4 = train_on([50, 25, 12, 6, 3])
_, train_loss5, test_loss5 = train_on([100, 200, 200, 100, 3])  # old model
_, train_loss5, test_loss5 = train_on([100, 200, 200, 100, 3])
x = np.arange(do_not_plot, epochs)
_, plt.plot(x, train_loss1_sig, "--", label="train [3,5,7,5,3] sigmoid")
_, plt.plot(x, train_loss1_ge, "--", label="train [3,5,7,5,3] gelu")
_, plt.plot(x, train_loss1_tanh, "--", label="train [3,5,7,5,3] tanh")

_, plt.plot(x, train_loss1, "--", label="train [3,5,7,5,3]")
_, plt.plot(x, train_loss2, "--", label="train [3,5,3]")
_, plt.plot(x, train_loss3, "--", label="train [50,25,12,25,50,3]")
_, plt.plot(x, train_loss4, "--", label="train [50,25,12,6,3]")
_, plt.plot(x, train_loss5, "--", label="train [100,200,200,100,3] - old model")
#
#
# plt.plot(x,test_loss1,label = 'test [3,5,7,5,3]')
# plt.plot(x,test_loss2,label = 'test [3,5,3]')
# plt.plot(x,test_loss3,label = 'test [50,25,12,25,50,3]')
# plt.plot(x,test_loss4,label = 'test [50,25,12,6,3]')
# plt.plot(x,test_loss5,label = 'test [100,200,200,100,3] - old model')
# after evaluate models [3,5,7,5,3] and [3,5,3] give realatively good result

model, _, _ = train_on([3, 5, 7, 5, 3], activation="gelu")

plt.xlabel("Loss")
plt.ylabel("Frequency")
plt.title(
    "Compare different configurations of NN with random_state ="
    + str(global_random_state)
)
plt.legend()
plt.show()


model.save(getFormatedTime() + "model.keras")
#
# time train model [3,5,7,5,3] is about 14% slower than [100,200,200,100,3] might due to https://chatgpt.com/c/66fb9e73-cfb8-8009-8c92-ae51ce3321e0
