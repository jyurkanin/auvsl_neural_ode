import matplotlib.pyplot as plt
import pandas as pd
import re

f = open("../build/train_output.txt")


training_loss = []
cv3_loss = []
ld3_loss = []
for line in f:
    if re.match("Avg Loss", line):
        training_loss.append(float(line.split()[2].replace(",","")))
    elif re.match("CV3 avg", line):
        cv3_loss.append(float(line.split()[3].replace(",","")))
    elif re.match("LD3 avg", line):
        ld3_loss.append(float(line.split()[3].replace(",","")))


plt.subplot(3,1,1)
plt.plot(training_loss)
plt.title("Training Loss")
plt.subplot(3,1,2)
plt.plot(cv3_loss)
plt.title("CV3 Loss")
plt.subplot(3,1,3)
plt.plot(ld3_loss)
plt.title("LD3 Loss")

plt.tight_layout()
plt.show()


