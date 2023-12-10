import matplotlib.pyplot as plt
import pandas as pd
import re
import argparse


parser = argparse.ArgumentParser(
                    prog='plot_loss',
                    description='This plots the loss',
                    epilog='bruh')

parser.add_argument('-f', '--filename')
args = parser.parse_args()

if args.filename == None:
    print("Need filename")
    quit()

f = open("../build/" + args.filename)


training_loss = []
validation = []
param0 = []

for line in f:
    if ("Avg Loss" in line) or ("Average" in line):
        training_loss.append(float(line.split()[2].replace(",","")))
    elif("validation" in line):
        validation.append(float(line.split()[3].replace(",","")))
    elif("Gradient" in line):
        param0.append(float(line.split()[4]))

plt.figure()
plt.plot(training_loss[0:30], color="k", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.title("Training Loss")

plt.figure()
plt.plot(validation[0:30], color="k", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
#plt.title("Validation Loss")

plt.figure()
plt.plot(param0, color="k")
plt.xlabel("Batch")
plt.ylabel("Gradient Norm")
#plt.title("Gradient Norm (dont plot)")

plt.tight_layout()
plt.show()


