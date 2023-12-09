import numpy as np


class Perceptron:
    def __init__(self,dim,learning_rate):
        self.dim=dim
        self.weights=np.random.randn(dim)
        self.biases=np.random.randn(1)
        self.learning_rate=learning_rate

    def weights_init(self):
        self.weights=0
        self.biases=0

    def forward(self,X):
        output= np.sign( np.dot(self.weights,X)+self.biases )

    def train(self,batch_X,batch_Y,epoch):
        Gram=batch_X.dot(batch_X.T)
        batch=batch_X.shape(0)
        for i in range(epoch):
            choose_data=np.random.randint(0,batch)
            output=batch_Y[choose_data]*()
