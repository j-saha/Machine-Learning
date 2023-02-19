import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import random
import time
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
        self.colors = []
        for kth in range(k):
            if(random.randint(0, 2) == 0):
                self.colors.append('red')
            if(random.randint(0, 2) == 1):
                self.colors.append('orange')
            else:
                self.colors.append('blue')

    def estimate_gmm(self):
        # plt.ion()
        for iter in range(iteration):
            self.E_step()
            self.M_step()
            self.plot_contour()

            if(self.check_likelihood()==False):
                break
            plt.clf()
        plt.ioff()
        # plt.show()
        # time.sleep(20000)
        # plt.ioff()


        print("Complete!")

        # plt.show(block=True)
        # plt.waitforbuttonpress()
        #


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
    def plot_contour(self):
        size = 1

        for kth in range(k):
            data_x, data_y, plot_means, plot_covs = self.run_PCA(kth)

            x = np.linspace(np.min(data_x) * size, np.max(data_x) * size, num=200)
            y = np.linspace(np.min(data_y) * size, np.max(data_y) * size, num=200)
            X, Y = np.meshgrid(x, y)
            XX = np.array([X.ravel(), Y.ravel()]).T
            Z = multivariate_normal.pdf(XX, mean=plot_means[kth], cov=plot_covs).reshape(X.shape)

            plt.ion()
            plt.scatter(data_x, data_y, 0.7)
            # plt.contour(X, Y, Z, colors=self.colors[kth])
            plt.contour(X, Y, Z)

            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.axis("tight")

            plt.title("Visualization of data points and gaussian distributions")

            plt.draw()
        plt.pause(0.2)


    def run_PCA(self, kth):
        if (self.data.shape[1] > 2):
            pca = PCA(n_components=2)
            plot_data = pca.fit_transform(self.data)
            plot_means = pca.fit_transform(self.means)
            plot_covs = np.dot(pca.components_, np.dot(self.covariance[kth], pca.components_.T))
            return plot_data[:, 0], plot_data[:, 1], plot_means, plot_covs
        else:
            return self.data[:, 0], self.data[:, 1], self.means, self.covariance[kth]



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
    iteration = 100
    threshold = 1e-4
    k2D = 3
    k3D = 4
    k6D = 5
    data = read_data(filename2D)
    k = k2D

    #ENTER CHOICE
    #------------------------------------------#

    choice = 2

    # ------------------------------------------#


    if(choice==3):
        data = read_data(filename3D)
        k = k3D
    elif(choice==6):
        data = read_data(filename6D)
        k = k6D
    gmm = GMM(k, iteration, threshold, data)
    x = gmm.estimate_gmm()
    plt.show()
