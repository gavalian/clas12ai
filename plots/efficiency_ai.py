import numpy as np
import matplotlib.pyplot as plt
import matplotlib

x = np.arange(14)
y = np.sin(x / 2)

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams["legend.loc"] = 'lower left'

plt.figure(figsize=(10,5))
plt.step(x, y + 2, label='pre (default)')
plt.plot(x, y + 2, 'C0o', alpha=0.5)

plt.step(x, y + 1, where='mid', label='negative tracks')
plt.plot(x, y + 1, 'C1o', alpha=0.5)

plt.step(x, y, where='post', label='post')
plt.plot(x, y, 'C2o', alpha=0.5)

plt.legend(title='Tracking Efficiency')
plt.show()
