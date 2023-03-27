import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 0001_CV_grass_GT.txt
# 0001_Tr_grass_GT.txt



# The main goal of this file is to resample all sources of data to 10hz

# Use this to obtain 
def readOdomFile(name, ii):
    fn = "/mnt/home/justin/Downloads/{0}/extracted_data/odometry/{1:04d}_odom_data.txt".format(name, ii)
    df = pd.read_csv(fn, names="cmd vel_left dist_left vel_right dist_right time".split())
    df = df["time vel_left vel_right".split()]
    return df

def readGTFile(name, ii):
    # unfortunate but whatever
    if(name == "CV3"):
        fn = "/mnt/home/justin/Downloads/{0}/localization_ground_truth/{1:04d}_CV_grass_GT.txt".format(name, ii)
    elif(name == "LD3"):
        fn = "/mnt/home/justin/Downloads/{0}/localization_ground_truth/{1:04d}_LD_grass_GT.txt".format(name, ii)
    elif(name == "Train3"):
        fn = "/mnt/home/justin/Downloads/{0}/localization_ground_truth/{1:04d}_Tr_grass_GT.txt".format(name, ii)
    else:
        print("You fucking fucked up you fuckign idiot")
        quit()
    
    df = pd.read_csv(fn, names = "time x y yaw".split())
    return df

def readIMUFile(name, ii):
    fn = "/mnt/home/justin/Downloads/{0}/extracted_data/imu/{1:04d}_imu_data.txt".format(name, ii)
    df = pd.read_csv(fn, names = "ax ay az wx wy wz qx qy qz qw time".split())
    df = df["time wx wy wz".split()]
    return df
    
def readFiles(name, num_files):
    timestep = .1
    
    for ii in range(1, num_files+1):
        df_odom = readOdomFile(name, ii)
        df_gt   = readGTFile(name, ii)
        df_imu  = readIMUFile(name, ii)
    
        start_time = df_gt["time"][0]
        end_time = df_gt["time"].iloc[-1]
    
        resample_times = np.arange(start_time, end_time, timestep)

        x_interp = np.interp(resample_times, df_gt["time"], df_gt["x"])
        y_interp = np.interp(resample_times, df_gt["time"], df_gt["y"])

        vx_interp = np.append(np.diff(x_interp), 0) / timestep # Finite differences
        vy_interp = np.append(np.diff(y_interp), 0) / timestep
        
        train_data = np.concatenate([resample_times.reshape(-1,1),
                                     np.interp(resample_times, df_odom["time"], df_odom["vel_left"]).reshape(-1,1),
                                     np.interp(resample_times, df_odom["time"], df_odom["vel_right"]).reshape(-1,1),
                                     x_interp.reshape(-1,1),
                                     y_interp.reshape(-1,1),
                                     (np.interp(resample_times, df_gt["time"], np.unwrap(df_gt["yaw"]))).reshape(-1,1),
                                     np.interp(resample_times, df_imu["time"], df_imu["wx"]).reshape(-1,1),
                                     np.interp(resample_times, df_imu["time"], df_imu["wy"]).reshape(-1,1),
                                     np.interp(resample_times, df_imu["time"], df_imu["wz"]).reshape(-1,1),
                                     vx_interp.reshape(-1,1),
                                     vy_interp.reshape(-1,1)
                                     ],
                                    axis=1)
        
        train_df = pd.DataFrame(train_data, columns="time vel_left vel_right x y yaw wx wy wz vx vy".split())
        train_df.to_csv("{0}_data{1:02d}.csv".format(name, ii))

def plotFile():
    df = pd.read_csv("train_data01.csv")
    plt.plot(df["time"], df["yaw"]) # % (2*np.pi)
    plt.show()

# Training dataset
readFiles("Train3", 17)
readFiles("CV3", 144)
readFiles("LD3", 1)



#plotFile()
