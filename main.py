"Server"

import os
import socket
from pathlib import Path

import pandas as pd

# định nghĩa host và port mà sever sẽ chạy và lắng nghe
HOST = "0.0.0.0"  # host ='localhost' will become '127.0.0.# 1' if IPv4 or '::1'if IPv6
PORT = 8000  # port = 4000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))

s.listen()  # chỉ chấp nhận 1 kết nối
print("Sever listening on port", PORT)

conn, addr = s.accept()
# conn.send("Connected")
print("Connected from", addr)

# sever sử dụng kết nối gửi dữ liệu tới client dưới dạng binary
import sys

print("first message: ")

s = conn.recv(1000)  # 96

import visualization

# import recordData
import numpy as np

# import pickle
from datetime import datetime

# from time import perf_counter
from tensorflow import keras
from collections import deque


def getFormatedTime():
    return datetime.now().strftime("%Y%m%d%H%M%S")


# file_path = "dataframe_"+formatted_time+".pkl"
def isInt(n):
    while n[-1] == " ":
        n = n[:-1]
    return n.isdigit()


def cls():
    import os

    os.system("cls")


batchSize = 61 * 4


L = []
firstProcess = 1
initialTime = 0


def func1(df: pd.DataFrame, newData: list):
    ndf = pd.DataFrame([L], columns=df.columns)
    df = pd.concat([df, ndf], axis=0, ignore_index=True)
    return df


currentPhoneTime = 0


def process():
    global firstProcess, initialTime, currentPhoneTime
    if L[-1] <= currentPhoneTime:
        return
    currentPhoneTime = L[-1]

    if firstProcess == 1:
        initialTime = int(L[-1])
        firstProcess = 0
    visualization.x_data.append(int(L[-1] - initialTime))
    y = np.array(L[:3])
    visualization.y_data.append(np.sum(y * y))
    y2 = np.abs(np.array(L[:3]))
    y2p = np.array(np.concatenate([y2, y2], axis=0)[1:4])
    visualization.y_data2.append(np.sum(y2 * y2p))

    y3p = np.array(np.concatenate([y2, y2], axis=0)[2:5])
    visualization.y_data3.append(np.sum(np.abs(y2)))
    visualization.update_plot()


SS = ""
df = pd.DataFrame(columns=["x", "y", "z", "time"])
firstTimeRecord = 1
startTimeRecord = 0
maxTimeRecord = 30

MODEL_PATH = Path(os.environ.get("LEGACY_ACTION_MODEL", "20240610183906model.keras"))
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Missing legacy Keras model. Train one with model.py or set "
        "LEGACY_ACTION_MODEL=/path/to/model.keras before running main.py."
    )

model = keras.models.load_model(MODEL_PATH)

Q = deque(maxlen=99)
for i in range(100):
    Q.append(0)

times = 0
while True:
    times += 1
    s = conn.recv(batchSize).decode("utf-8")
    # print("s:",s)
    SS += s
    if len(SS) > 80:
        s = SS[:80]
        # print(s)
        SS = SS[80:]
        L = list(map(float, s.split()))
        L[0] -= 0.26
        L[1] -= 0.05
        L[2] -= 9.739
        arr = np.array(L[:-1])
        Sum = np.sum(arr * arr)
        Q.append(Sum)
        if times % 100 == 0:
            X = np.array(Q)
            X = X.reshape(1, 99)
            visualization.idTitle = np.argmax(model.predict(X).reshape(-1))
            print(visualization.titles[visualization.idTitle])

    # break by irregular received data
    if len(s) == 0:
        break
# visualization.plt.show()
