import numpy as np
if __name__ == '__main__':
    print('PyCharm')
    batch_X=np.array([[1,2,3],[4,5,6],[7,8,9]])
    batch_y=np.array([[1],[-1],[1]])
    Gram = batch_X.dot(batch_X.T)
    print(Gram)