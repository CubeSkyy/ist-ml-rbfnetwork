import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
class RBFNetwork:

    def __init__(self, initc, rate, b, w, x, sigma, target, max_epochs, max_epochs_kmeans):
        self.c = kmeans(x, initc, max_epochs_kmeans)[0]
        self.rate = rate
        self.b = b
        self.w = w
        self.x = x
        self.sigma = sigma
        self.target = target
        self.max_epochs = max_epochs

    def RBF(self,x, c, sigm):
        return np.exp(-(np.linalg.norm(x - c) ** 2) / 2 * (sigm ** 2))


    def logistic(self,x):
        return np.exp(x) / (1 + np.exp(x))


    def getRBFS(self,x):
        rbfs = np.empty(self.c.shape[0])
        for i, ci in enumerate(self.c):
            rbf = self.RBF(x, ci, self.sigma)
            rbfs[i] = rbf
            print("RBF", i, ":", rbf)
        return rbfs


    def query(self,x):
        print("\n---------------")
        print("QUERY: Point:", x, "\n")
        rbfs = self.getRBFS(x)

        hidden_output = np.inner(rbfs, self.w) + self.b
        print("Output of hidden layer:", hidden_output)

        output = self.logistic(hidden_output)
        print("Output of final layer:", output)

        print("Predicted Target:", 0 if output > 0.5 else 1)


    def train(self):

        for k in range(self.max_epochs):
            print("\n------------------------------------------")
            print("Epoch:", k)
            old_w = self.w
            for j, ele in enumerate(self.x):
                print("\n---------------")
                print("Point", j, ":", ele, "\n")
                rbfs = self.getRBFS(ele)

                hidden_output = np.inner(rbfs, self.w) + self.b
                print("Output of hidden layer:", hidden_output)

                output = self.logistic(hidden_output)
                print("Output of final layer:", output)

                print("Predicted Target:", 0 if output > 0.5 else 1)
                print("Target:", self.target[j])

                diff = (self.target[j] - output)

                self.w = self.w + self.rate * (diff * rbfs)
                print("New Weights:", self.w)

                self.b = self.b + diff
                print("New bias:", self.b)
                print("\n---------------")

            if(np.allclose(self.w, old_w)):
                print("\n------------------------------------------")
                print("Algorithm converged in", k, "steps.")
                print("Weights:", self.w)
                print("Bias:", self.b)
                print("------------------------------------------")
                break

def main():
    initc = np.array([[0, 0], [-1, -0]])
    max_epochs_kmeans = 2
    rate = 1
    b = 1
    w = np.array([1, 1])
    x = np.array([[0, 0], [0, -1], [-1, 0], [-1, -1]],dtype=float)
    sigma = 1
    target = np.array([1, 0, 0, 1])
    max_epochs = 3

    rbfNetwork = RBFNetwork(initc, rate, b, w, x, sigma, target, max_epochs, max_epochs_kmeans)
    rbfNetwork.train()
    rbfNetwork.query(np.array([0, 0]))
    rbfNetwork.query(np.array([-1, 0]))
    rbfNetwork.query(np.array([0, -1]))
    rbfNetwork.query(np.array([-1, -1]))

if __name__ == '__main__':
    main()
