import matplotlib.pyplot as plt
import pandas as pd
import re

f = open("../build/train_output.txt")


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

plt.subplot(3,1,1)
plt.plot(training_loss, marker='2')
plt.title("Training Loss")

plt.subplot(3,1,2)
plt.plot(validation, marker='2')
plt.title("Validation Loss")

plt.subplot(3,1,3)
plt.plot(param0, marker='2')
plt.title("Param[0] Loss")

plt.tight_layout()
plt.show()


