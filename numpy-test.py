import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

print(str(a.ndim) + "//" + str(a.shape) + "//" + str(len(a)))

print(np.arange(10))
print(np.arange(1, 9, 2))
print(np.linspace(0, 1, 6))

print(np.ones((3, 3)))
print(np.eye(3))

print(np.diag(np.array([1, 2, 3, 4])))

print(np.random.rand(4)) #uniform
print(np.random.randn(4)) #Guassian
