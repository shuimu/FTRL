import sys
import math

"""
intruductions
1. input data split by \t
"""

## parameters
alpha = 0.1
beta = 1
lambda1 = 1
lambda2 = 1

## data info
D = 3
Z = [0] * D
N = [0] * D
W = [0] * D

def sgn(k):
    if k>0:
        return 1
    elif k==0:
        return 0
    else:
        return -1

def get_w(i):
    if abs(Z[i]) <= lambda1:
        return 0
    else:
        return -1.0/( (beta+math.sqrt(N[i]))/alpha + lambda2) * (Z[i] - sgn(Z[i])*lambda1)

def sigmoid(k):
    return 1.0/(1+math.exp(-k))


def predict(X):
    sum = 0
    for i in range(D):
        sum += X[i]*W[i]
    return sigmoid(sum)

def FTRL():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        items = map(float, line.strip().split('\t'))
        features = [1] + items[1:]
        label = items[0]
        I = set()
        for i in range(len(features)):
            if features[i] != 0:
                I.add(i)
        for i in I:
            W[i] = get_w(i)
        p = predict(features)
        for i in I:
            g = (p - label) * features[i]
            delta = (math.sqrt(g**2+N[i])-math.sqrt(N[i]))/alpha
            Z[i] = Z[i] + g - delta*W[i]
            N[i] = N[i] + g**2
    for i in range(D):
        print "%d\t%.3f"%(i,W[i])
        

FTRL()

