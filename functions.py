import numpy as np

def optimalLSR(X, y):
    xTx = np.dot(X.T, X)
    xTy = np.dot(X.T, y)
    return np.dot(np.linalg.pinv(xTx), xTy)

def optimalSolution(X, y, alpha):
    I = np.identity(X.shape[1])
    n = X.shape[0]
    xTx = np.dot(X.T, X)
    xTy = np.dot(X.T, y)
    return np.dot(np.linalg.pinv(xTx + n * alpha * I), xTy)

def computeER(X, y, w):
    n = X.shape[0]
    return np.sqrt(1/(n) * np.linalg.norm(np.dot(X, w) - y)**2)

def computeErrorPointEstimate(X, y, w):
    n = X.shape[0]
    return np.sqrt(1/(n) * np.linalg.norm(w - y)**2)