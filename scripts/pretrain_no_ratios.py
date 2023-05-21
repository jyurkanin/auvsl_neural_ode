import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

global count
#import pdb; pdb.set_trace() #this is completely overpowered. Too useful.


def print_c_network(md, in_mean, in_std, output_scaler):
  torch.set_printoptions(threshold=10000)
  output = "//#include \"solver.h\"\n//Auto Generated by pretrain.py\n\nvoid Solver::load_nn_gc_model(){\n"
  for c in md.keys():
    temp = str(c)
    name = temp[2:] + temp[0]
    output += ((str(md[c].flatten()).replace("tensor([", name + " << ").replace("])", ";")) + "\n")
  
  #output += "out_mean " + str(output_scaler.mean_.tolist()).replace("[","<< ").replace("]", ";") + "\n"
  output += "out_std " + str(np.sqrt(output_scaler.var_).tolist()).replace("[","<< ").replace("]", ";") + "\n"
  
  output += "in_mean " + str(in_mean.tolist()).replace("[","<< ").replace("]", ";") + "\n"
  output += "in_std " + str(in_std.tolist()).replace("[","<< ").replace("]", ";") + "\n"
  
  output += "}"
  with open('solver_nn_gc.cpp','w') as f:
    f.write(output)
  return


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Current compute device:", device)

#These are determined to be the best parameters by checking all possible and removing the one that are unimportant.
in_features = "vx vy w zr".split()
out_features = "Fx Fy Fz".split()

df = pd.read_csv("tire_data.csv")

data_x = np.array(df[in_features])
data_y = np.array(df[out_features])

# print("Mean: ", np.mean(temp_data_y, axis=0))
# print("Std: ", np.std(temp_data_y, axis=0))

# select_rows = np.all(np.abs(temp_data_y - np.mean(temp_data_y, axis=0)) < 1.0 * np.std(temp_data_y, axis=0), axis=1)
# print(select_rows.shape)
# data_x = temp_data_x[select_rows]
# data_y = temp_data_y[select_rows]

# print("Shapes:")
# print(temp_data_x.shape, data_x.shape)
# print(temp_data_y.shape, data_y.shape)

data_len = data_x.shape[0]

train_data = data_x[:int(data_len*.95),:]
label_data = data_y[:int(data_len*.95),:]

test_data = data_x[int(data_len*.95):,:]
test_labels = data_y[int(data_len*.95):,:]

#input_scaler = StandardScaler()
output_scaler = StandardScaler(with_mean=False)

#input_scaler.fit(train_data)
label_data = output_scaler.fit_transform(label_data)

#print("input_scaler:", input_scaler.mean_)

# plt.scatter(train_data[:,1], label_data[:,1])
# plt.title("Vy vs Fy")
# plt.show()

# plt.scatter((train_data[:,2]*.098) - train_data[:,0], label_data[:,0])
# plt.title("Vx vs Fx")
# plt.show()


train_data = torch.from_numpy(train_data).float()
label_data = torch.from_numpy(label_data).float()

train_data = train_data.to(device)
label_data = label_data.to(device)

#import pdb; pdb.set_trace() #this is completely overpowered. Too useful.
#10 hidden nodes, 2 layers seems best.

class TireNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.in_size = 4 # sinkage, qd, vx, vy
    self.hidden_size = 32
    self.out_size = 3
    
    self.tire_radius = .098
    
    self.loss_fn = torch.nn.MSELoss()
    self.model = nn.Sequential(
      nn.Linear(self.in_size, self.hidden_size),
      nn.Tanh(),
      nn.Linear(self.hidden_size, self.hidden_size),
      nn.Tanh(),
      nn.Linear(self.hidden_size, self.out_size),
      nn.ReLU()
    )

  def compute_bekker_input_scaler(self, x):
    tire_tangent_vel = x[:,2]*self.tire_radius
    diff = tire_tangent_vel - x[:,0]
    slip_lon = torch.abs(diff)
    slip_lat = torch.abs(x[:,1])
    tire_abs = torch.abs(x[:,2])
    bekker_args = torch.cat((x[:,3][:,None], # zr
                             slip_lon[:,None], # diff
                             tire_abs[:,None], # |qd|
                             slip_lat[:,None], # vy
                             ), 1)
    
    self.in_mean = torch.mean(bekker_args, 0)
    temp = bekker_args - self.in_mean
    self.in_std = torch.sqrt(torch.var(temp, 0))
    
  # vx,vy,qd,zr 5 bekker params, x is (9, batch_size)
  def forward(self, x):
    tire_tangent_vel = x[:,2]*self.tire_radius
    diff = tire_tangent_vel - x[:,0]
    slip_lon = torch.abs(diff)
    slip_lat = torch.abs(x[:,1])
    tire_abs = torch.abs(x[:,2])
    bekker_args = torch.cat((x[:,3][:,None],   # zr
                             slip_lon[:,None], # diff
                             tire_abs[:,None], # |qd|
                             slip_lat[:,None], # vy
                             ), 1)
    
    bekker_args = (bekker_args - self.in_mean) / self.in_std
    
    yhat = self.model.forward(bekker_args)
    
    yhat_sign_corrected = torch.cat((
      (yhat[:,0] * torch.tanh(1*diff))[:,None],
      (yhat[:,1] * torch.tanh(-1*x[:,1]))[:,None],
      (yhat[:,2] / (1 + torch.exp(-1*x[:,3])))[:,None]), 1)
    return yhat_sign_corrected
    
model = TireNet()
model.compute_bekker_input_scaler(train_data)
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-1)

count = 0
def fit(lr, batch_size, epochs):
    global count
    
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    
    for j in range(epochs):
        for i in range(0, train_data.shape[0]-batch_size, batch_size):
            x = train_data[i:i+batch_size, :]
            y = label_data[i:i+batch_size, :]
            
            opt.zero_grad()
            y_hat = model.forward(x)
            loss = model.loss_fn(y_hat, y)
            loss.backward()
            opt.step()
        plt.scatter(count, loss.item(), color='b')
        count += 1
        print("LOSS", loss)



def get_evaluation_loss(test_x, test_y):
    #input_vec = input_scaler.transform(test_x)
    input_vec = test_x
    predicted_force = model.forward(torch.from_numpy(input_vec).float()).detach().numpy()
    predicted_force = output_scaler.inverse_transform(predicted_force)
    print(np.mean(np.square(predicted_force.flatten() - test_y.flatten())))

    plt.title("Vx vs Fx")
    #plt.scatter(test_x[:100,0], test_y[:100,0])
    #plt.scatter(test_x[:100,0], predicted_force[:100,0])
    plt.scatter(test_x[:100,0], predicted_force[:100,0] - test_y[:100,0])
    plt.show()

    plt.title("Vy vs Fy")
    #plt.scatter(test_x[:100,1], test_y[:100,1])
    #plt.scatter(test_x[:100,1], predicted_force[:100,1])
    plt.scatter(test_x[:100,1], predicted_force[:100,1] - test_y[:100,1])
    plt.show()

    plt.title("zr vs Fz")
    #plt.scatter(test_x[:100,3], test_y[:100,2])
    #plt.scatter(test_x[:100,3], predicted_force[:100,2])
    plt.scatter(test_x[:100,3], predicted_force[:100,2] - test_y[:100,2])
    plt.show()


def fx_plot():
    test = np.zeros((1000,4))
    test[:,0] = 0                      # vx
    test[:,1] = 0;                     # vy
    test[:,2] = np.linspace(-5,5,1000) # diff
    test[:,3] = 0.001                  # zr
    
    #test_norm = input_scaler.transform(test)
    test_norm = test
    x = torch.from_numpy(test_norm)
    test_out = model.forward(x.float()).detach().numpy()
    test_out = output_scaler.inverse_transform(test_out)[:,0]
    
    plt.plot((test[:,2]*.098) - test[:,0], test_out)
    plt.title("Slip Ratio vs. Longitudinal Force (Fx)")
    plt.xlabel("Slip Ratio")
    plt.ylabel("Longitudinal Force (N)")
    plt.show()    

def fy_plot():
    test = np.zeros((1000,9))
    test[:,0] = 0.2                    # vx
    test[:,1] = np.linspace(-1,1,1000) # vy
    test[:,2] = 0.1                    # w
    test[:,3] = 0.001                  # zr
    
    #test_norm = input_scaler.transform(test)
    test_norm = test
    x = torch.from_numpy(test_norm)
    test_out = model.forward(x.float()).detach().numpy()
    test_out = output_scaler.inverse_transform(test_out)[:,1]
    
    plt.plot(test[:,1], test_out)
    plt.title("Slip Angle vs. Lateral Force (Fy)")
    plt.xlabel("Slip Angle")
    plt.ylabel("Lateral Force (N)")
    plt.show()

def fz_plot():
    test = np.zeros((1000,9))
    test[:,0] = 0.2                    # vx
    test[:,1] = 0                      # vy
    test[:,2] = 0.1                    # w
    test[:,3] = np.linspace(-.1,.1,1000) # zr
    
    #test_norm = input_scaler.transform(test)
    test_norm = test
    x = torch.from_numpy(test_norm)
    test_out = model.forward(x.float()).detach().numpy()
    test_out = output_scaler.inverse_transform(test_out)[:,2]
    
    plt.plot(test[:,3], test_out)
    plt.title("Sinkage vs. Normal Force (Fy)")
    plt.xlabel("Sinakge Angle")
    plt.ylabel("Normal Force (N)")
    plt.show()

    

model_name = "train_no_ratio2.net"
md = torch.load(model_name)
model.load_state_dict(md)

fit(1e-3, 5000, 100)
plt.show()
get_evaluation_loss(test_data, test_labels)
fit(1e-3, 50, 10)
plt.show()

md = model.state_dict()
print_c_network(md, model.in_mean, model.in_std, output_scaler)
torch.save(md, model_name)

get_evaluation_loss(test_data, test_labels)
fx_plot()
fy_plot()
# fz_plot()



