# Use the linear model to generate fake data

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

vehicle_model = np.array([[ 0.0472,  0.0438],
                          [ 0.0036, -0.0035],
                          [-0.1744,  0.1748]])

def saveTrajectory(filename, vl, vr, duration):
    print("Simulating Filename:", filename)
    
    xk = np.zeros((3,1)) # [x; y; theta]

    dt = .001
    num_samples = 100*10

    # Time, vl, vr, x, y, yaw, wx, wy, wz, vx, vy
    train_data = np.zeros((num_samples, 11))
    vel = np.zeros((2,1))
    dx = np.zeros((3,1))
    
    # 10s. 100 steps of .01 times 10s
    for j in range(num_samples):
        train_data[j, 0] = j / 100.0
        train_data[j, 1] = vl
        train_data[j, 2] = vr
        train_data[j, 3] = xk[0]  # x
        train_data[j, 4] = xk[1]  # y
        train_data[j, 5] = xk[2]  # yaw
        train_data[j, 6] = 0      # wx
        train_data[j, 7] = 0      # wy
        train_data[j, 8] = dx[2]  # wz
        train_data[j, 9] = vel[0]  # vx
        train_data[j, 10] = vel[1] # vy

        for i in range(10):
            u = np.array([[vl],[vr]])
            dx = np.dot(vehicle_model, u)
            
            rot = np.array([np.cos(xk[2]), -np.sin(xk[2]), np.sin(xk[2]),  np.cos(xk[2])]).reshape((2,2))
            vel = np.dot(rot, dx[0:2])

            xk[0] += dt*vel[0]
            xk[1] += dt*vel[1]
            xk[2] += dt*dx[2]
        
        xk[2] = np.arctan2(np.sin(xk[2]), np.cos(xk[2]))
            
    # plt.plot(train_data[:,3], train_data[:,4])
    # plt.quiver(train_data[::100,3], train_data[::100,4], train_data[::100,9], train_data[::100,10], scale=1.0, color="red")
    # plt.quiver(train_data[::100,3], train_data[::100,4], np.cos(train_data[::100,5]), np.sin(train_data[::100,5]), scale=10.0, color="green")
    # plt.show()
    train_df = pd.DataFrame(train_data, columns="time vel_left vel_right x y yaw wx wy wz vx vy".split())
    train_df.to_csv(filename)


saveTrajectory("Train3_data18.csv", -8, 8, 6.1)
saveTrajectory("Train3_data19.csv", -4, 4, 6.1)
saveTrajectory("Train3_data20.csv", -2, 2, 6.1)
saveTrajectory("Train3_data21.csv", 1, 2, 6.1)
saveTrajectory("Train3_data22.csv", 3, 4, 6.1)
saveTrajectory("Train3_data23.csv", 7, 8, 6.1)
saveTrajectory("Train3_data24.csv", 11, 12, 6.1)
saveTrajectory("Train3_data25.csv", 2, 4, 6.1)
saveTrajectory("Train3_data26.csv", 6, 8, 6.1)
saveTrajectory("Train3_data27.csv", 10, 12, 6.1)
saveTrajectory("Train3_data28.csv", 1, 4, 6.1)
saveTrajectory("Train3_data29.csv", 5, 8, 6.1)
saveTrajectory("Train3_data30.csv", 9, 12, 6.1)
saveTrajectory("Train3_data31.csv", 2, 2, 6.1)
saveTrajectory("Train3_data32.csv", 4, 4, 6.1)
saveTrajectory("Train3_data33.csv", 6, 6, 6.1)
saveTrajectory("Train3_data34.csv", 8, 8, 6.1)
saveTrajectory("Train3_data35.csv", 10, 10, 6.1)
saveTrajectory("Train3_data36.csv", 12, 12, 6.1)
