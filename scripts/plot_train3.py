import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 0001_Tr_grass_GT.txt



def plot_train3():
    for i in range(1,18):
        fn = "Train3_data{0:02d}.csv".format(i)
        df = pd.read_csv(fn)
        print("Time:", df["time"].iloc[-1] - df["time"].iloc[0])
        # plt.plot(df["x"], df["y"])
        # plt.show()
    
    
    
plot_train3()
