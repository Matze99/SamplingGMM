# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt

def sample_norm():
    d = np.random.uniform(0, 1, size=(1000, 100000))
    d = np.mean(d, axis=1)
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def normal(value):
    return 1/np.sqrt(2*np.pi)* np.exp(-value**2/2)

def gaussian(value, mean, variance):
    return 1/(np.sqrt(2*np.pi)*variance) * np.exp(-1/(2*variance**2)*(value-mean)**2)

def inv_sigmoid(value):
    return np.log(value/(1-value))

def plot_sample_uniform_gaussian_inverse_map():
    fig = plt.figure()
    x = np.linspace(-5, 5, 201)
    y = normal(x)

    y_s = np.random.uniform(0,1, size=1000)
    x_s = inv_sigmoid(y_s)

    n, bins, patches = plt.hist(x=x_s, bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='sigmoid map', density=True)
    scale = 1#np.max(n) * np.sqrt(2 * np.pi)
    plt.plot(x, y*scale, label='normal distribution')
    plt.title('sampling normal distribution')
    plt.legend()
    plt.ylabel('p(x)')
    plt.xlabel('x')

    plt.show()
    fig.savefig('../../Portfolio/sigmoid_map_vs_normal.png')

def plot_monte_carlo_inv_sigmoid_map():
    fig = plt.figure()
    x = np.linspace(-5, 5, 201)
    y = normal(x)

    size = 2000000
    y_s = np.random.uniform(0, 1, size=size)
    x_s = inv_sigmoid(y_s)
    y_samp = np.random.uniform(0, 1, size=size)
    y_top = normal(x_s)
    index_filtered = filter(lambda x: True if y_top[x] >= y_samp[x] else False, list(range(len(y_s))))
    x_filtered = [x_s[i] for i in index_filtered]
    print(len(x_filtered))

    n, bins, patches = plt.hist(x=x_filtered, bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='monte carlo', density=True)
    scale = 1  # np.max(n) * np.sqrt(2 * np.pi)
    plt.plot(x, y * scale, label='normal distribution')
    plt.title('monte carlo sampling normal distribution')
    plt.legend()
    plt.ylabel('p(x)')
    plt.xlabel('x')

    plt.show()
    # fig.savefig('../../Portfolio/sigmoid_map_monte_carlo.png')

def plot_central_limit_theorem_sigmoid_map():
    fig = plt.figure()
    x = np.linspace(-5, 5, 201)
    y = normal(x)

    y_s = np.random.uniform(0, 1, size=(2000, 3))
    x_s = np.mean(inv_sigmoid(y_s), axis=1)

    n, bins, patches = plt.hist(x=x_s, bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='sampling mean', density=True)
    scale = 1  # np.max(n) * np.sqrt(2 * np.pi)
    plt.plot(x, y * scale, label='normal distribution')
    plt.title('central limit theorem')
    plt.legend()
    plt.ylabel('p(x)')
    plt.xlabel('x')

    plt.show()
    # fig.savefig('../../Portfolio/sigmoid_map_central_limit.png')

def plot_central_limit_theorem_sigmoid_map_non_normal(mean=0, variance=1):
    fig = plt.figure()
    x = np.linspace(-5*variance+mean, 5*variance+mean, 201)
    y = gaussian(x, mean, variance)

    y_s = np.random.uniform(0, 1, size=(2000, 3))
    x_s = np.mean(inv_sigmoid(y_s), axis=1)*variance+mean

    n, bins, patches = plt.hist(x=x_s, bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='sampling mean', density=True)
    scale = 1  # np.max(n) * np.sqrt(2 * np.pi)
    plt.plot(x, y * scale, label='gaussian distribution')
    plt.title(f'Gaussian with mean {mean} and variance {variance}')
    plt.legend()
    plt.ylabel('p(x)')
    plt.xlabel('x')

    plt.show()
    fig.savefig('../../Portfolio/shifted_gaussian.png')

def plot_central_limit_vs_monte_carlo():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle('Central Limit vs Monte Carlo')

    x = np.linspace(-5, 5, 201)
    y = normal(x)

    size = 2000000
    y_s = np.random.uniform(0, 1, size=size)
    x_s = inv_sigmoid(y_s)
    y_samp = np.random.uniform(0, 1, size=size)
    y_top = normal(x_s)
    index_filtered = filter(lambda x: True if y_top[x] >= y_samp[x] else False, list(range(len(y_s))))
    x_filtered = [x_s[i] for i in index_filtered]
    print(len(x_filtered))

    n, bins, patches = ax2.hist(x=x_filtered, bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='monte carlo', density=True)

    y_s = np.random.uniform(0, 1, size=(2000000, 3))
    x_s = np.mean(inv_sigmoid(y_s), axis=1)

    n, bins, patches = ax1.hist(x=x_s, bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='sampling mean', density=True)

    scale = 1  # np.max(n) * np.sqrt(2 * np.pi)
    ax1.plot(x, y * scale, label='normal distribution')
    ax2.plot(x, y * scale, label='normal distribution')
    # plt.title('monte carlo sampling normal distribution')
    # plt.legend()
    ax1.set_ylabel('p(x)')
    ax1.set_xlabel('x')
    ax2.set_xlabel('x')
    plt.show()
    fig.savefig('../../Portfolio/monte_carlo_vs_central_limit.png')

def plot_norm_plus_sample():
    x = np.linspace(-5,5,201)
    y = normal(x)

    y_s = np.random.uniform(-1000, 1000, size=(201, 100000))
    x_s = np.mean(y_s, axis=1)


    n, bins, patches = plt.hist(x=x_s, bins='auto', color='r',
                                alpha=0.7, rwidth=0.4, label='sigmoid sampled')

    y_s = np.random.uniform(0, 1, size=201)
    x_s = inv_sigmoid(y_s)

    n, bins, patches = plt.hist(x=x_s, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.4, label='uniform sample mean')

    y_s_e = np.random.uniform(0, 1, size=(201, 5))
    x_s_e = np.mean(inv_sigmoid(y_s_e), axis=1)

    n, bins, patches = plt.hist(x=x_s_e, bins='auto', color='g',
                                alpha=1, rwidth=0.6, label='uniform sigmoid sample mean')

    scale = np.max(n)*np.sqrt(2*np.pi)
    plt.plot(x, y* scale, label='normal distribution')
    plt.title('sampling normal distribution')
    plt.legend()
    plt.ylabel('p(x)')
    plt.xlabel('x')
    plt.show()

def multivariate_gaussian(values, mean, covariance):
    fac = 1/np.sqrt((2*np.pi)**len(covariance)*np.linalg.det(covariance))
    return np.diagonal(fac*np.exp((values-mean) @ np.linalg.inv(covariance) @ (values-mean).T))

def plot_multivariate_gaussian():
    mean = np.array([0,0])
    # covariance = np.array([[1, 2], [2, 1]])
    covariance = np.array([[1, 0], [0,1]])

    lambda_, gamma_ = np.linalg.eig(covariance)

    fig = plt.figure()
    # x1 = np.linspace(-2.5, 5, 100).reshape((-1, 1))
    # x2 = np.linspace(-15, 32, 100).reshape((-1, 1))
    # y = multivariate_gaussian(np.concatenate((x1, x2),1), mean, covariance).reshape((-1,1))
    # plt.contourf(x1, x2, y)

    y_s = np.random.uniform(0, 1, size=(1000, 3))
    x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, 2))
    x_s = (x_normal*lambda_) @ gamma_ + mean

    # n, bins, patches = plt.hist(x=x_s, bins='auto', color='r',
    #                             alpha=0.7, rwidth=0.85, label='sampling mean', density=True)
    scale = 1  # np.max(n) * np.sqrt(2 * np.pi)
    # plt.plot(x, y * scale, label='gaussian distribution')
    plt.scatter(x_s[:, 0], x_s[:, 1])
    plt.title(f'Multivariate Gaussian')
    # plt.legend()
    plt.ylabel('x2')
    plt.xlabel('x1')

    plt.show()
    # fig.savefig('../../Portfolio/shifted_gaussian.png')


def plot_gaussian_mixture_model():
    mus = [np.array([0, 1]), np.array([20, 1]), np.array([-3, 3])]
    covs = [np.array([[1, 2], [2, 1]]), np.array([[1, 0], [0, 1]]), np.array([[10, 1], [1, 0.3]])]

    # mus = [np.array([0, 1]), np.array([7, 10]), np.array([-3, -3])]
    # covs = [np.array([[1, 2], [2, 1]]), np.array([[1, 0], [0, 1]]), np.array([[2, 1], [1, 0.3]])]

    pis = np.array([0.3, 0.1, 0.6])
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]
    assert np.isclose(acc_pis[-1], 1)

    n = 1000
    samples = []

    for i in range(n):
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
        y_s = np.random.uniform(0, 1, size=(dimensions * 1, 3))
        x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, dimensions))
        # transforming into multivariate distribution
        x_multi = (x_normal * lambda_) @ gamma_ + selected_mu
        samples.append(x_multi.tolist()[0])

    fig = plt.figure()
    samples = np.array(samples)
    plt.scatter(samples[:, 0], samples[:, 1])
    plt.title('Gaussian Mixture Model Samples')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.show()
    fig.savefig('../../Portfolio/samples_gmm_2.png')

def gmm(value, pis, means, variances):
    density = np.zeros(value.shape)
    for i, mean in enumerate(means):
        density += pis[i]/np.sqrt(2*np.pi)/variances[i] * np.exp(-1/(2*variances[i]**2)*(value-mean)**2)
    return density

def plot_1D_gaussian_mixture_model():
    means = [-2, 4]
    variances = [2,1]
    pis = [0.3, 0.7]
    acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis) + 1)]

    n = 1000
    samples = []

    for i in range(n):
        # sample uniform
        r = np.random.uniform(0, 1)
        # select gaussian
        k = 0
        for i, threshold in enumerate(acc_pis):
            if r < threshold:
                k = i
                break

        selected_mu = means[k]
        selected_cov = variances[k]

        # sampling from normal distribution
        y_s = np.random.uniform(0, 1, size=(1, 3))
        x_normal = np.mean(inv_sigmoid(y_s), axis=1).reshape((-1, 1))
        # transforming into multivariate distribution
        x_multi = (x_normal) * selected_cov + selected_mu
        samples.append(x_multi.tolist()[0])

    x_cal = np.linspace(-10, 10, 200)
    y_cal = gmm(x_cal, pis, means, variances)

    fig = plt.figure()

    n, bins, patches = plt.hist(x=np.array(samples), bins='auto', color='r',
                                alpha=0.7, rwidth=0.85, label='sampled gmm', density=True)
    plt.plot(x_cal, y_cal, label='gmm')
    plt.title('Sampled 1D gaussian mixture model')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()
    plt.show()
    fig.savefig('../../Portfolio/1D_gmm_sample.png')




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    plot_gaussian_mixture_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
