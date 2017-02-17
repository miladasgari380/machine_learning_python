import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
print(iris['data'].shape)

# extracting data
samples = iris['data']

# extracting class types
classes = iris['target']

# train-test creation 75% VS 25%
X_train, X_test, y_train, y_test = train_test_split(samples, classes, random_state=1234)
print(X_train.shape)
print(X_test.shape)

# pair plot of features
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
plt.suptitle("iris_pairplot")
for i in range(3):
    for j in range(3):
        ax[i, j].scatter(X_train[:, j], X_train[:, i+1], c=y_train)
        ax[i, j].set_xticks(())
        ax[i, j].set_yticks(())

        if(i == 2):
            ax[i, j].set_xlabel(iris['feature_names'][j])
        if(j == 0):
            ax[i, j].set_ylabel(iris['feature_names'][i + 1])
        if(j > i):
            ax[i, j].set_visible(False)
# plt.show()

# KNN
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print(knn)

# make prediction using this model
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print(str(prediction[0])+": which means: "+str(iris['target_names'][prediction]))

# Evaluation
y_pred = knn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
# Or
accuracy = knn.score(X_test, y_test)
print(accuracy)
