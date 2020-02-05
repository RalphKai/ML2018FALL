import numpy as np
import pandas as pd
from matplotlib.mlab import PCA
# X = [[1,1,1],[1,2,1],[1,3,2],[1,4,3]]

X = [[1,2,3],[4,8,5],[3,12,9],[1,8,5],[5,14,2],[7,4,1],[9,8,9],[3,8,1],[11,5,6],[10,11,7]]
X = np.array(X)
mean = np.mean(X, 0)
print(mean)
# ans = PCA(np.array(X))
# print('ans:', ans)

C = X - mean
# print(C)

cov = np.dot(C.T, C)
cov = (cov) / (len(X) - 1)
print('covariance matrix:\n', cov)

eval, evec = np.linalg.eig(cov)
print('\neval:\n', eval)
print('\nevec:\n', evec)

index = np.argsort(eval)

index = index[::-1]
# print(index)

PCA_diag = np.diag(eval[index])
PCA_axis = evec[:,index]
print('\nsorted_eigenvalue:\n', PCA_diag)
print('\nsorted_eigenvector:\n', PCA_axis)

PCA_axis_2D = PCA_axis[:, :2]
print('\nPCA_axis_2D:\n', PCA_axis_2D)
PCA_components_back = PCA_axis.T.dot(X.T)
PCA_components = PCA_axis_2D.T.dot(X.T)
print('\nPCA_components\n', PCA_components.T)
print('\nPCA_components_back\n', PCA_components_back.T)
PCA_reconstruction = PCA_axis_2D.dot(PCA_components).T
PCA_back = PCA_axis.dot(PCA_components_back).T
print('\nPCA_reconstruction\n', PCA_reconstruction)
print('\nPCA_back\n', PCA_back)

gap = X - PCA_reconstruction
print('\ngap:\n', gap)
gap_back = X - PCA_back
print('\ngap_back:\n', gap_back)

dist = []
for i in range(X.shape[0]):
    dist.append( np.sqrt(np.sum(np.dot(gap[i,:].T, gap[i,:]))) )

dist = np.array(dist)
print('\nError:\n', dist)
# X = np.array(X).T
# print(X)
# cov = np.cov(X)

# df = pd.DataFrame([(1, 2, 3), (4, 8, 5), (3,12,9), (1,8,5), (5,14,2),(7,4,1),(9,8,9),(3,8,1), (11,5,6),(10,11,7)],columns=['dogs', 'cats', 'frogs'])
# print(df.cov())
# print(cov)