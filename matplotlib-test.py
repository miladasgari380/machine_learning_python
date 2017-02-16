#http://www.scipy-lectures.org/intro/matplotlib/index.html#matplotlib

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3, 20)
y = np.linspace(0, 9, 20)

plt.plot(x, y)
plt.plot(x, y, '+')
plt.show()

image = np.random.rand(30, 30)
plt.imshow(image, cmap=plt.cm.hot)
plt.colorbar()
plt.show()


n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)
T = np.arctan2(Y, X) #angle between X, Y
plt.axes([0.025, 0.025, 0.95, 0.95])
plt.scatter(X, Y, s=75, c=T, alpha=.5)
plt.xlim(-1.5, 1.5)
plt.xticks(())
plt.ylim(-1.5, 1.5)
plt.yticks(())

plt.show()
