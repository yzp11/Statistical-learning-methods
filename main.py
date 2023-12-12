import numpy as np
from perceptron import Perceptron
if __name__ == '__main__':
    batch_X=np.array([[-1,4,3],[4,0,9],[7,-3,-40],[-8,0,-9],[5,5,2],[5,-242,342],[-42,-4,-2],[12,3,-43],[-4,34,2],[-43,2,4],[-43,-2,-4]])
    batch_Y=np.array([-1,1,1,-1,1,1,-1,1,-1,-1,-1])
    X=np.array([[-7,9,-5]])
    perceptron=Perceptron(3,0.1,batch_X,batch_Y)
    perceptron.train(50)
    print(perceptron.forward(X))

