import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib
import sys

def readFile(filename):
    Z = np.random.rand(6, 112)
    file = open(filename,'r')
    for i in range(0,6):
        line = file.readline()
        size = len(line)
        for w in range(0,112):
            if line[w]=='X':
                Z[i][w] = 3.0
    return Z


matplotlib.rcParams["figure.figsize"] = [12.4, 4.8]
matplotlib.rcParams.update({'font.size': 4})

#Z = np.random.rand(6, 112)
#Z[1][1] = 3.0
#Z[2][2] = 3.0
#Z[3][3] = 3.0
#Z[4][4] = 3.0
Z = readFile(sys.argv[1])
print(Z)
#plt.figure(figsize=(12,6))
fig, axis = plt.subplots(6, 1)

#c = ax0.pcolor(Z)
#ax0.set_title('default: no edges')

#c = axis.pcolor(Z, edgecolors='k', linewidths=1)
for i in range(0,6):
    Z = readFile(sys.argv[i+1])
    c = axis[i].pcolor(Z, edgecolors='#222266', linewidths=1)
    axis[i].set_ylabel('SL - ' + str(i))
#axis[0].set_title('drift chambe event')
#axis[0].set_xlabel('wire #')
#axis[0].set_ylabel('drift chamber layer')


fig.tight_layout()
#plt.savefig('a.pdf')
plt.show()
