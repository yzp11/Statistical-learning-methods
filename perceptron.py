import numpy as np
class Perceptron:
    def __init__(self,dim,learning_rate,batch_X,batch_Y):
        self.dim=dim
        self.batch = batch_X.shape[0]
        self.weights=np.random.randn(self.batch)
        self.biases=np.random.randn(1)
        self.learning_rate=learning_rate
        self.batch_X=batch_X
        self.batch_Y=batch_Y
        self.Gram=batch_X.dot(batch_X.T)


    def weights_init(self):
        self.weights=0
        self.biases=0

    def forward(self,X):
        temp=np.zeros(self.dim)
        for i in range(self.batch):
            print()
            temp+=(self.weights[i]*self.batch_Y[i])*self.batch_X[i]
        output= np.sign( np.dot(temp,X.T)+self.biases )
        return output

    def train(self,epoch):
        for i in range(epoch):
            choose_data=np.random.randint(0,self.batch)
            temp=0
            for j in range(self.batch):
                temp+=self.weights[j]*self.batch_Y[j]*self.Gram[j][choose_data]
            if self.batch_Y[choose_data]*(temp+self.biases)<=0:
                self.weights[choose_data]+= self.learning_rate
                self.biases+= self.learning_rate*self.batch_Y[choose_data]
