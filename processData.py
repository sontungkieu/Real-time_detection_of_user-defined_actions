import numpy as np
import pandas as pd

# thử dùng raw data

df = pd.read_csv("20240509141447.csv")

# print(df.head())
df["time"] -= min(df["time"])
df["time"] /= 9
df["f1"] = df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2
L = df["f1"].values
LL = []
for i in range(0, len(L) - 100, 5):
    LL.append(L[i : i + 100])
arr = np.array(LL)
arr = np.concatenate((arr, np.ones((arr.shape[0], 1))), axis=1)
tmp = arr[:, -1]
tmp *= 2
df = pd.DataFrame(arr)
print(df)
print(arr.shape)

df.to_csv("train2.csv", index=False)

"/////////////////////////////////////////////////////////////////////////////////////"
# ndf = pd.DataFrame()
# ndf['time'] = df['time']
# L=df['time'].values
# for i in range(0,len(L)):L[i]=round(L[i])
# df['time'] = L
# print(df['time'].values)
# ndf['f1'] = df['f1']
# # print(df.head())
# # print(df.loc[1].values)
# arr = df.values
# arr = arr[:,1:]
# print(arr[-1,-1])
# print(len(arr))
# print(arr)
# # cnt = np.ndarray([100])
# # for i in range(1,len(arr)):
# #     print(arr[i,-1]-arr[i-1,-1])
# # print(cnt)
