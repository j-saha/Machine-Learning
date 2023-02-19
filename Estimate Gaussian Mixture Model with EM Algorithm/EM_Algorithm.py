import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings("ignore")
class GMM:
    def __init__(self, k, iteration, threshold, data):
        print(k)
        self.k = k
        self.iteration = iteration
        self.threshold = threshold
        self.log_likelihood = -np.inf
        self.n = data.shape[0]
        self.m = data.shape[1]
        self.means = np.random.rand(self.k, self.m)
        self.covariance = []
        self.w = [1.0/self.k]*k
        for i in range(k):
            self.covariance.append(np.identity(self.m))
        self.epsilon = 1e-6 * np.identity(self.m)
        self.data = data
        self.probabilities = np.zeros((self.n, self.k))
        self.temp_probabilities = np.zeros((self.n, self.k))
    def estimate_gmm(self):
        for iter in range(iteration):
            self.E_step()
            self.M_step()
            if(self.check_likelihood()==False):
                return
        return

    def E_step(self):
        for i in range(k):
            pdf = multivariate_normal.pdf(self.data, mean=self.means[i], cov=self.covariance[i], allow_singular=True)
            self.probabilities[:,i] = self.w[i] * pdf
        sum = self.probabilities.sum(axis=1, keepdims=True)
        self.probabilities = self.probabilities/sum

    def M_step(self):
        for i in range(k):
            prob_data = self.probabilities[:, i, None] * data
            self.means[i] = prob_data.sum(axis=0) / self.probabilities[:, i].sum()
            self.covariance[i] = np.cov(data.T, aweights=self.probabilities[:, i], bias=True) + self.epsilon
            self.w[i] = self.probabilities[:, i].sum() / self.n
            self.temp_probabilities[:, i] = self.w[i] * multivariate_normal.pdf(self.data, mean=self.means[i],
                                                                                cov=self.covariance[i],
                                                                                allow_singular=True)
    def check_likelihood(self):
        temp_log_likelihood = np.log(self.temp_probabilities.sum(axis=1)).sum()
        if np.abs(self.log_likelihood - temp_log_likelihood) < self.threshold:
            return False
        else:
            self.log_likelihood = temp_log_likelihood
            return True


def read_data(filename):
    return np.loadtxt(filename)

def plot(gmms):
    ks = []
    log_likelihoods = []
    for gmm in gmms:
        ks.append(gmm.k)
        log_likelihoods.append(gmm.log_likelihood)
    plt.plot(ks, log_likelihoods)
    plt.xlabel("Component Number: k")
    plt.ylabel("Log likelihood")
    plt.title("k vs Log Likelihood Graph")
    plt.show()




if __name__ == '__main__':
    filename2D = "data2D.txt";filename3D = "data3D.txt";filename6D = "data6D.txt"
    data = read_data(filename2D)
    iteration = 100
    threshold = 1e-4
    gmms = []
    for k in range(1, 11):
        gmm = GMM(k, iteration, threshold, data)
        gmm.estimate_gmm()
        gmms.append(gmm)
    plot(gmms)