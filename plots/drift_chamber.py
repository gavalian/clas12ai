import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib

matplotlib.rcParams["figure.figsize"] = [12.4, 4.8]
matplotlib.rcParams.update({'font.size': 14})

#Z = np.random.rand(36, 112)

Z = np.zeros(shape=(36,112))
for x in range(0,36):
    for y in range(0,112):
        Z[x][y] = 0.25

Z[1][1] = 1
Z[2][2] = 1
Z[3][3] = 1
Z[4][4] = 1
Z[0][0] = 0

print(Z)
#plt.figure(figsize=(12,6))
fig, axis = plt.subplots(1, 1)

#c = ax0.pcolor(Z)
#ax0.set_title('default: no edges')

#c = axis.pcolor(Z, edgecolors='k', linewidths=1)
c = axis.pcolor(Z, edgecolors='#222266', linewidths=1)
axis.set_title('drift chambe event')
axis.set_xlabel('wire #')
axis.set_ylabel('drift chamber layer')

fig.tight_layout()
#plt.savefig('a.pdf')
plt.show()
