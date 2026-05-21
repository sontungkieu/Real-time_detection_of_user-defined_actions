import matplotlib.pyplot as plt

# import main
import numpy as np
import random
from collections import deque

mx = 100
# Initialize empty lists to store data
x_data = deque(maxlen=mx)
y_data = deque(maxlen=mx)
y_data2 = deque(maxlen=mx)
y_data3 = deque(maxlen=mx)

x_data.append(0)
y_data.append(0)
y_data2.append(0)
y_data3.append(0)
idTitle = 0
titles = ["do nothing", "jogging", "run"]


# Create a function to update the plot
def update_plot():
    plt.xlim(x_data[0], x_data[-1])
    # print(x_data[0], x_data[-1])
    # print(x_data)
    plt.cla()  # Clear the current plot
    plt.plot(x_data, y_data, color="blue")
    plt.plot(x_data, y_data2, color="red")
    plt.plot(x_data, y_data3, color="purple")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Real-time Data Plot:" + titles[idTitle])
    plt.pause(0.01)  # Pause to update the plot


# Create an empty plot
plt.figure()

# Update the plot whenever the update function is called
# update_plot()

# Call the update function whenever needed
# For example:
# while True:  # Update the plot 10 times
#     update_plot()
#
# plt.show()
