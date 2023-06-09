import matplotlib.pyplot as plt
import pandas as pd
import re

f = open("../build/avg.txt")


training_loss = []
cv3_loss = []
ld3_loss = []
param0 = []
for line in f:
    if re.match("Avg Loss", line):
        training_loss.append(float(line.split()[2].replace(",","")))
    elif re.match("CV3 avg", line):
        cv3_loss.append(float(line.split()[3].replace(",","")))
    elif re.match("LD3 avg", line):
        ld3_loss.append(float(line.split()[3].replace(",","")))
    elif re.match("Param norm", line):
        param0.append(float(line.split()[4].replace(",","")))


plt.subplot(4,1,1)
plt.plot(training_loss, marker='o')
#plt.yscale(value="log")
plt.title("Training Loss")

plt.subplot(4,1,2)
plt.plot(cv3_loss, marker='o')
#plt.yscale(value="log")
plt.title("CV3 Loss")

plt.subplot(4,1,3)
plt.plot(ld3_loss, marker='o')
#plt.yscale(value="log")
plt.title("LD3 Loss")

plt.subplot(4,1,4)
plt.plot(param0, marker='o')
#plt.yscale(value="log")
plt.title("Param[0]")

plt.tight_layout()
plt.show()


