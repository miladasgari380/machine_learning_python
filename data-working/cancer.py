import mglearn as mglearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(cancer.keys())
print(cancer.data.shape)

from sklearn.datasets import load_boston

boston = load_boston()
print(boston.keys())
print(boston.data.shape)

X, y = mglearn.datasets.load_extended_boston()
print(X.shape)

# mglearn.plots.plot_knn_classification(n_neighbors=2)
# plt.title("one neighbor")
# plt.show()

from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_forge()
# print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier

# ----------- overfitting phenamenon in KNN classification ------------ #
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(X_train, y_train)
#
# # decision boundary
# fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# # zip for making tuples
# for n_neighbors, ax in zip([1, 3, 9], axes):
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=60, cmap=mglearn.cm2)
#     ax.set_title("%d neighbors" % n_neighbors)
# # plt.show()

# ----------------------------------------------------------------------

# ---------- working with breast cancer data ---------- #
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=66)
train_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    train_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, train_accuracy, label="train accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.legend()
# plt.show()
# -----------------------------------------------------

# ----------- overfitting phenamenon in KNN regression ------------ #
from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X_train, y_train)
    ax.plot(X, y, 'o')
    ax.plot(X, -3 * np.ones(len(X)), 'o')
    ax.plot(line, reg.predict(line))
plt.show()
# -----------------------------------------------------------------

