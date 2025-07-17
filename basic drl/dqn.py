import numpy as np

class DQN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        self.lr = lr
        self.w1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim)
        self.b2 = np.zeros(output_dim)

    def predict(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        return self.z2 

    def train(self, x, y_target):
        y_pred = self.predict(x)

        dloss =  (y_pred - y_target) * (2 / x.shape[0])

        dw2 = np.dot(self.a1.T, dloss)
        db2 = np.sum(dloss)

        da1 = np.dot(dloss, self.w2.T)
        dz1 = da1 * (1 - np.tanh(self.z1) ** 2)

        dw1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        self.w1 -= self.lr * dw1
        self.b1 -= self.lr * db1
        self.w2 -= self.lr * dw2
        self.b2 -= self.lr * db2