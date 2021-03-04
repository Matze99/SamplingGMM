# Code Block 1: Sampling from a normal distribution
import numpy as np

def inv_sigmoid(values):
    return np.log(values/(1-values))

number_of_samples = 2000

y_s = np.random.uniform(0, 1, size=(number_of_samples, 3))
x_s = np.mean(inv_sigmoid(y_s), axis=1)

variance, mean = 5, 30
x_arbitrary = np.mean(inv_sigmoid(y_s), axis=1)*variance+mean

# Code Block 2

# eigenvalue decomposition
lambda_, gamma_ = np.linalg.eig(np.array([[1, 2], [2, 1]]))

dimensions = len(lambda_)
# sampling from normal distribution
y_s = np.random.uniform(0, 1, size=(dimensions*1000, 3))
x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
# transforming into multivariate distribution
x_multi = (x_normal*lambda_) @ gamma_ + mean

# Code Block 3

mus = [np.array([0, 1]), np.array([7, 1]), np.array([-3, 3])]
covs = [np.array([[1, 2], [2, 1]]), np.array([[1, 0], [0, 1]]), np.array([[10, 1], [1, 0.3]])]

pis = np.array([0.3, 0.1, 0.6])
acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis)+1)]
assert np.isclose(acc_pis[-1], 1)

# sample uniform
r = np.random.uniform(0, 1)
# select gaussian
k = 0
for i, threshold in enumerate(acc_pis):
    if r < threshold:
        k = i
        break

selected_mu = mus[k]
selected_cov = covs[k]

# sample from selected gaussian
lambda_, gamma_ = np.linalg.eig(selected_cov)

dimensions = len(lambda_)
# sampling from normal distribution
y_s = np.random.uniform(0, 1, size=(dimensions*1, 3))
x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
# transforming into multivariate distribution
x_multi = (x_normal*lambda_) @ gamma_ + selected_mu

