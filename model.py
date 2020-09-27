#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jacobstachowicz
"""
import numpy as np
import numpy.matlib as npm
import pickle as p
import random
import os
import matplotlib.pyplot as plt
import math

directory = r"C:\[insert path]\code\cifar-10-batches-py" # for windows
#directory = "/[insert path]/code/cifar-10-batches-py"   # for mac/linux


def paramChange(params, new, i):
    stuff    = [np.copy(i) for i in params]
    stuff[i] = new
    return stuff     

def computeNumGrads(X, Y, params, gamma, beta, lmbda, val, batchNorm):
    grads = [np.copy(i) for i in params]
    gGrads = [np.copy(i) for i in gamma]
    bGrads = [np.copy(i) for i in beta]
    c     = cost(X, Y, params, gamma, beta, lmbda, batchNorm)

    for i in range(int(len(params)/2)):
        W = params[i]
        for k in range(np.size(W, 0)):
            for j in range(W.shape[1]):
                Wi = np.array(W)
                Wi[k,j] += val
                cj = cost(X, Y, paramChange(params, Wi, i), gamma, beta, lmbda,batchNorm)
                grads[i][k,j] = (cj - c) / val
    
    for i in range(int(len(grads)/2),len(params)):
        b = params[i]
        for k in range(np.size(b)):
            bi = np.array(b)
            bi[k] += val
            ck = cost(X, Y, paramChange(params, bi, i), gamma, beta, lmbda, batchNorm)
            grads[i][k] = (ck - c) / val

    if(batchNorm):

        for i in range(len(gamma)):
            W = gamma[i]
            for k in range(np.size(W, 0)):
                for j in range(W.shape[1]):
                    Wi = np.array(W)
                    Wi[k,j] += val
                    cj = cost(X, Y, params, paramChange(gamma, Wi, i), beta, lmbda, batchNorm)
                    gGrads[i][k,j] = (cj - c) / val
            
        for i in range(len(bGrads)):
            b = bGrads[i]
            for k in range(np.size(b)):
                bi = np.array(b)
                bi[k] += val
                ck = cost(X, Y, params, gamma, paramChange(beta, bi, i), lmbda,batchNorm)
                bGrads[i][k] = (ck - c) / val
   
    return grads, gGrads, bGrads


def plotLoss(cost_train, cost_valid, yLabel):
    
    eta_min, eta_max, n_s, t_end, cycles = etas
     
    plt.figure(dpi=120)
    plt.plot(cost_train, "-g", label="training "+ str(yLabel))
    plt.plot(cost_valid, "-r", label="validation" + str(yLabel))
    plt.ylabel(yLabel)
    plt.xlabel('update step')
    plt.legend(loc="lower right")
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.grid(True)

    max_x       = int(math.ceil(t_end / 1000.0)) * 1000
    perCycle    = 10
    cycle_len   = n_s*2
    step        = int((cycle_len)/perCycle)
    array_len   = len(cost_valid)
    my_ticks    = np.arange(0, max_x, step)
    frequency   = 10
    
    plt.xticks(np.arange(0, array_len+1, frequency), my_ticks[::frequency])
    plt.show()  
    
    
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = p.load(fo, encoding='bytes')
    return dict


def fetch_batches(type_of_batch):
    arr = []
    path = os.fsencode(directory)
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.startswith(type_of_batch):
            arr.append(unpickle("cifar-10-batches-py/" + filename))
    return arr


def largestBatch():
    data = fetch_batches("data_batch")
    X, Y, y    = loadBatch(data[0])
    validSize = 10
    for i in range(1, len(data)):
        newX, newY, newy    = loadBatch(data[i])
        X = np.concatenate((X, newX), axis=0)
        Y = np.concatenate((Y, newY), axis=0)
        y = np.concatenate((y, newy), axis=0)
        
    Xtrain = np.delete(X, np.s_[len(X)-validSize:len(X)], axis=0)
    Ytrain = np.delete(Y, np.s_[len(Y)-validSize:len(Y)], axis=0)
    ytrain = y[0:len(y)-validSize]
    Xvalid = np.delete(X, np.s_[0:len(X)-validSize], axis=0)
    Yvalid  = np.delete(Y, np.s_[0:len(Y)-validSize], axis=0)
    yvalid  = y[len(y)-validSize:len(y)]
    return Xtrain, Ytrain, ytrain, Xvalid , Yvalid , yvalid 


def emptyLists(amount):
    arr = []
    for i in range(amount):
        arr.append([])
    return arr


def fixedLists(amount, size):
    arr = []
    for i in range(amount):
        arr.append([None] * size)
    return arr


def hot_ones(labels):
    hot_list = np.zeros( (len(labels), 10))
    for i in range(len(labels)):
        hot_list[i][labels[i]] = 1
    return hot_list
    

def loadBatch(batch):
    X = np.asarray(batch[b'data'])
    y = np.asarray(batch[b'labels'])
    Y = hot_ones(y)
    return X, Y, y


def multiT(X,Y,Xv,Yv):
    return X.T, Y.T, Xv.T, Yv.T


def createMiniBatch(X, Y, y, size):
    newX = np.zeros([len(X[:,0]), size])
    newY = np.zeros( (len(Y), size))
    newy = np.zeros([size])
    a = np.arange(len(y))
    random.shuffle(a)
    
    for i in range(size):
        newX[:,i] = X[:,a[i]] 
        newY[:,i] = Y[:,a[i]] 
        newy[i] = y[a[i]]
        
    return newX, newY, newy


def createMiniBatches(X, Y, y, n_batches):
    size = int(np.size(X, 1)/n_batches)
    batch_array = [None] * n_batches
    a = np.arange(np.size(X, 1))
    random.shuffle(a)
    for ii in range(int(n_batches)):
        step = ii * size
        newY = np.zeros((np.size(Y, 0), size))
        newX = np.zeros([np.size(X, 0), size])
        newy = np.zeros([size])
        for i in range(size):
            newX[:,i] = X[:,a[i+step]] 
            newY[:,i] = Y[:,a[i+step]] 
            newy[i]   = y[a[i+step]]
        b_dict =	{
            "X": newX,
            "Y": newY,
            "y": newy
        }
        batch_array[ii] = b_dict
    return batch_array


def unpackMiniBatch(b, i):
    return b[i]["X"], b[i]["Y"], b[i]["y"]
    

def preprocess(X):
    X = X - npm.repmat(np.mean(X, axis = 0), len(X), 1)
    return np.divide(X, npm.repmat(np.std(X, axis = 0), len(X), 1))


def createParams(info):
    Ws = []
    bs = []
    for i in range(len(info)-1):
        Ws.append(np.random.randn(info[i+1], info[i]) * np.sqrt(2/info[i] ) )
        bs.append(np.zeros((info[i+1], 1)))
    return Ws + bs


def createParamsSensitive(info, sig):
    Ws = []
    bs = []
    for i in range(len(info)-1):
        
        Ws.append(np.random.normal(0, sig, (info[i+1], info[i]) ) )
        bs.append(np.zeros((info[i+1], 1)))
    return Ws + bs


def createYB(info):
    Y = []
    B = []
    for i in range(len(info)-1):
        Y.append(np.ones([info[i+1], 1]))
        B.append(np.zeros([info[i+1], 1]))
        
    return Y, B

def leakyReLu(s):
    return np.maximum(s*0.1, 0)

def reLu(s):
    return np.maximum(s, 0)


def softMax(s):
    return np.exp(s) / np.sum(np.exp(s), axis = 0)


def batchNormalise(S, mu, v):
    eps = 1e-9
    return ( S - mu ) / np.sqrt(eps + v)


def forward(*args):
    mu, v, x, s, s_hat = emptyLists(5)
    X, params, gamma, beta, batchNorm = args[:5]
    testing = len(args) == 7
    if(testing):
        mu, v = args[-2:]
    k = int(len(params) / 2)
    x.append(np.copy(X))    
    
    for i in range(k - 1):
        s.append(np.matmul(params[i], x[i]) + params[k + i])        #5
        if(batchNorm):
            if(not testing):
                mu.append(r(np.mean(s[i], axis = 1)))
                v.append( r(np.var(s[i],  axis = 1)))

            s_hat.append(batchNormalise(s[i], mu[i], v[i]))         #6
            s_tmp = np.multiply(gamma[i], s_hat[i]) + beta[i]       #7
            x.append(reLu(s_tmp))                                   #8
        else:
            x.append(reLu(s[i]))              

    s_last = np.matmul(params[k-1], x[k-1]) + params[2*k -1] 
    p  = softMax(s_last) 
    return p, x, s, s_hat, mu, v


def crossEntropyLoss(P, Y):
    lcrosssum = 0;
    for i in range(np.size(Y,1)):
        lcrosssum -= np.log(np.dot(Y[:,i], P[:,i]))
    return lcrosssum


def regularization(params, lmbda):
    return lmbda *  np.sum([np.sum(params[i] * params[i]) for i in range(int(len(params)/2))]) 


def cost(X, Y, weightsAndBias, gamma, beta, lmbda, batchNorm):
    f = forward(X, weightsAndBias, gamma, beta, batchNorm)
    lcrosssum = crossEntropyLoss(f[0], Y)
    reg = regularization(weightsAndBias, lmbda)
    return (1/np.size(X, 1)) * lcrosssum + reg


def ComputeAcc(X, y, params, gamma, beta, mu, v, batchNorm):
    P = forward(X, params, gamma, beta, batchNorm, mu, v)[0]
    corr = 0
    tot = len(y)
    for i in range(tot):
        corr += 1 if y[i] == np.argmax(P[:,i]) else 0
    return round(100*(corr/tot),2)


def ComputeEnsambleAcc(X, y, nets, batchNorm):
    params, gamma, beta, mu, v = nets[0]

    P = forward(X, params, gamma, beta, batchNorm, mu, v)[0]

    for n in range(len(nets)-1):
        params, gamma, beta, mu, v = nets[n+1]
        Pt = forward(X, params, gamma, beta, batchNorm, mu, v)[0]
        P += Pt

    corr = 0
    tot = len(y)
    for i in range(tot):
        corr += 1 if y[i] == np.argmax(P[:,i]) else 0
    return round(100*(corr/tot),2)

def printAcc(Xo, yo, Xv, yv,params, gamma, beta, mu, v,batchNorm ):
    print("post train acc: "+str(ComputeAcc(Xo, yo, params, gamma, beta, mu, v,batchNorm)) + " %")
    print("post valid acc: "+str(ComputeAcc(Xv, yv, params, gamma, beta, mu, v,batchNorm)) + " %")


def r(n):
    return np.reshape(n, (len(n), 1))


def toParams(W1, W2, b1, b2):
    return W1, W2, b1, b2


def cLearnRates(etas, t):
    eta_min, eta_max, n_s, i, cycles = etas
    l = int(t / (n_s*2))
    if(t % (2*n_s) < n_s): 
        return eta_min + ( (t-(2*l * n_s)) /n_s) * (eta_max - eta_min) 
    else: 
        return eta_max - ( (t-((2*l+1) * n_s)) /n_s) * (eta_max - eta_min) 
    
    
def batch_norm_back_pass(G, S, mu, v, div):
    eps = 1e-5

    sigma_1 = np.power((v+eps), -0.5) # (1/(np.sqrt(v+eps)))                #31
    sigma_2 = np.power((v+eps), -1.5) #(1/((v * np.sqrt(v))+eps))           #32
    
    G1 = np.multiply(G, sigma_1)                                            #33
    G2 = np.multiply(G, sigma_2)                                            #34
    
    D = S - mu                                                              #35
    c = r(np.sum(np.multiply(G2, D), axis = 1))                             #36

    part1 = div * r( np.sum(G1, axis=1))
    part2 = div * np.multiply(D, r(np.sum(c, axis = 1)))
    
    return  G1 - part1 - part2
    

def backwards(X, Y, params, g, b, lmbda, batchNorm):
    k = int(len(params) / 2)
    div = (1/np.size(Y,1))
    
    P, h, s, s_hat, mu, v = forward(X, params, g, b, batchNorm)
    
    grads = [np.copy(i) for i in params]
    gGrads = np.copy(g)
    bGrads = np.copy(b)

    G = P - Y                                                                           #21
    grads[k-1] = div * np.matmul(G, np.transpose(h[k-1])) + 2 * lmbda * params[k-1]     #22 ekv 1
    grads[2*k-1] = r(div * np.sum(G, axis=1))                                           #22 ekv 2
    
    G = np.matmul(np.transpose(params[k-1]), G)                                         #23
    G = np.multiply(G, (h[k-1] > 0))                                                    #24
    
    for i in range(k-2, -1, -1):
        if(batchNorm):
            gGrads[i] = r(div * np.sum(np.multiply(G, s_hat[i]), axis = 1))             #25 ekv 1
            bGrads[i] = r(div * np.sum(G, axis=1))                                      #25 ekv 2
        
            G = np.multiply(G, r(np.sum(g[i], axis=1)))                                 #26
            G = batch_norm_back_pass(G, s[i], mu[i], v[i], div )                        #27
        
        grads[i] = div * np.matmul(G, np.transpose(h[i])) + 2 * lmbda * params[i]
        grads[i+k] = r(div * np.sum(G, axis=1))
    
        if(i > 0):
            G = np.matmul(np.transpose(params[i]), G)
            G = np.multiply(G, (h[i] > 0)) 

    return grads, gGrads, bGrads, mu, v
    

def miniBatchGD(G, n_batches, etas, params, gamma, beta, l, batchNorm, printStuff = False):
    Xo, Yo, yo, Xv, Yv, yv = G
    Xt, Yt, yt             = test
    
    eta_min, eta_max, n_s, t_end, cycles = etas
    
    stepPrint = int(2*n_s)/10
    max_x     = (int(math.ceil(t_end / 1000.0)) * 1000)
    size      = int(max_x/stepPrint)+1

    cost_t, cost_v, loss_t, loss_v, acc_t, acc_v, t_rate = fixedLists(7, size)
    batches = createMiniBatches(Xo, Yo, yo, n_batches)
    ii = 0
    mu_avg = 0.0
    v_avg  = 0.0
    alpha = 0.9
    for t in range(t_end):
           
        eta = cLearnRates(etas, t)
        if(t % n_batches == 0):
            batches = createMiniBatches(Xo, Yo, yo, n_batches)
        i = t % n_batches
        Xb, Yb, yb = unpackMiniBatch(batches, i)
        
        grads, gGrads, bGrads, mu, v = backwards(Xb, Yb, params, gamma, beta, l, batchNorm)
        
        for i in range(len(params)):
            params[i] = params[i] - eta * grads[i]
        
        if(batchNorm):
            for i in range(len(gamma)):
                gamma[i] = gamma[i] - eta * gGrads[i]
                beta[i]  = beta[i]  - eta * bGrads[i]
            if(t ==0):
                mu_avg = mu
                v_avg = v
            for i in range(len(mu_avg)):
                mu_avg[i] = mu_avg[i]*alpha + (1.0-alpha)*mu[i]
                v_avg[i] = v_avg[i]*alpha + (1.0-alpha)*v[i]
            
        if(t% stepPrint==0 or t == t_end-1):
            if((t% (stepPrint*4)==0) and printStuff):
                print(str(100 * (t/t_end)) + " Procent done")
            cost_t[ii]  = cost(Xo, Yo, params, gamma, beta, l, batchNorm)
            cost_v[ii]  = cost(Xv, Yv, params,gamma, beta, l, batchNorm)
            
            loss_t[ii] = cost(Xo, Yo, params, gamma, beta,0, batchNorm)
            loss_v[ii] = cost(Xv, Yv, params,gamma, beta, 0, batchNorm)
        
            acc_t[ii]   = ComputeAcc(Xo, yo, params, gamma, beta, mu_avg, v_avg, batchNorm)
            acc_v[ii]   = ComputeAcc(Xv, yv, params, gamma, beta, mu_avg, v_avg, batchNorm)
            if(printStuff):
                printAcc(Xo, yo, Xv, yv,params, gamma, beta, mu_avg, v_avg ,batchNorm )
                print("TestACC: " + str(ComputeAcc(Xt, yt, params, gamma, beta, mu_avg, v_avg, batchNorm)))
            
            t_rate[ii] = eta

            ii += 1
    final_v_acc = ComputeAcc(Xv, yv, params, gamma, beta, mu_avg, v_avg,batchNorm)
    if(printStuff):
        plotLoss(cost_t, cost_v, "Cost")
        plotLoss(loss_t, loss_v, "Loss")
        plotLoss(acc_t, acc_v, "Accuracy")
        plotLoss(t_rate, t_rate, "ting rate")

        printAcc(Xo, yo, Xv, yv,params, gamma, beta, mu_avg, v_avg, batchNorm )
        Xt, Yt, yt = test
        print("test acc: " + str(ComputeAcc(Xt, yt, params, gamma, beta, mu_avg, v_avg,batchNorm)))
    return params, gamma, beta,mu_avg, v_avg, final_v_acc


def logLambda():
    e_min = -4
    e_max = -1.5
    l = e_min + (e_max - e_min) * random.random()
    return np.power(10, l);


def logLambdaOrdered(i):
    e_min = -2.5
    e_max = -2.15 #-1.5
    l = e_min + (e_max - e_min) * i
    return np.power(10, l);


def bestParams(G, n_batches, etas, weightsAndBias, gamma, beta, l, test, batchNorm, times):

    eta_min, eta_max, n_s, t_end, cycles = etas
    testX, testY, testy = test
    
    final_valid_acc = 0 
    second_valid_acc = 0 
    
    final_lmbda = lmbda
    s_final_lmbda = final_lmbda

    final_params = [np.copy(i) for i in weightsAndBias]
    final_gamma = np.copy(gamma)
    final_beta = np.copy(beta)
    final_mu = None
    final_v = None
    
    s_final_params = [np.copy(i) for i in weightsAndBias]
    s_final_gamma = np.copy(gamma)
    s_final_beta = np.copy(beta)
    s_final_mu = None
    s_final_v = None
    
    for i in range(times):
        new_lmbda = logLambda()
        print(i/times)
        print("search nr: " + str(i+1) + " with lambda: "  + str(new_lmbda))
        p = [np.copy(i) for i in weightsAndBias]
        g = np.copy(gamma)
        b = np.copy(beta)
        new_params, new_gamma, new_beta,mu,v, v_acc = miniBatchGD(G, n_batches, etas, p ,g, b,new_lmbda, batchNorm)
        print("post vali acc:  " + str(v_acc))
        print("post test acc:  " + str(ComputeAcc(testX, testy, new_params, new_gamma, new_beta, mu, v, batchNorm)))
        if(v_acc > final_valid_acc):
            s_final_lmbda = final_lmbda
            s_final_params = final_params
            s_final_gamma = final_gamma
            s_final_beta = final_beta
            s_final_mu = final_mu
            s_final_v = v
            second_valid_acc = final_valid_acc
            
            final_lmbda = new_lmbda
            final_params = new_params
            final_gamma = new_gamma
            final_beta = new_beta
            final_mu = mu
            final_v = v
            final_valid_acc = v_acc
        ComputeAcc(testX, testy, new_params, new_gamma, new_beta, mu, v, batchNorm)
    test_acc =ComputeAcc(testX, testy, final_params, final_gamma,final_beta, final_mu, final_v, batchNorm)
    s_test_acc =ComputeAcc(testX, testy, s_final_params, s_final_gamma,s_final_beta, s_final_mu, s_final_v, batchNorm)
    print("##################")
    print("best lambda : " + str(final_lmbda))
    print("best valid acc: " + str(final_valid_acc))
    print("gave test acc: " + str(test_acc))
    print("------------------")
    print("2nd best lambda : " + str(s_final_lmbda))
    print("gave valid acc: " + str(second_valid_acc))
    print("gave test acc: " + str(s_test_acc))
        

def gradNorm(aGrad,nGrad,name):
    print("Norm_" + name + ": " + str(np.linalg.norm(aGrad - nGrad) ))


def relativeError(aGrad,nGrad, name):
    eps   = 1e-05
    upper = np.linalg.norm(aGrad - nGrad) 
    lower = max(eps, np.linalg.norm(aGrad) + np.linalg.norm(nGrad))
    print("relative error for " + name + ": " + str(upper/lower))
    

def testGradients(X, Y, y, batchsize, weightsAndBias, gamma, beta,lmbda,batchNorm):
    miniX, miniY, miniy                    = createMiniBatch(X, Y, y, batchsize)

    aGrads, gGrads, bGrads, mu, v = backwards(miniX, miniY, weightsAndBias,gamma,beta, lmbda, batchNorm)
    nGrads, ngGrads, nbGrads = computeNumGrads(miniX, miniY, weightsAndBias, gamma, beta, lmbda, 1e-5,batchNorm)
    
    netLen = len(weightsAndBias)
    lenG = len(gGrads)
    for i in range(netLen):
        name = "W"+str(i+1)
        if i >= int(netLen/2):
            name = "b"+str(i+1 - int(netLen/2))
        relativeError(aGrads[i],nGrads[i],name)

    if(batchNorm):
        for i in range(lenG):
            relativeError(gGrads[i],ngGrads[i],"Gamma")
            relativeError(bGrads[i],nbGrads[i],"Beta")
        
    
################################  Exercise 3 Bonus   ################################

for i in range(1):
    
    testX, testY, testy    = loadBatch(unpickle(directory + "/test_batch"))
    
    trainX, trainY, trainy, validX, validY, vaildy= largestBatch() # training set of size 49000 and test set of size 1000
    trainX = preprocess(trainX)
    validX = preprocess(validX)
    testX  = preprocess(testX)
    
    """ Used for reducing dimensionality (use only when testing gradients)"""
#    trainX = np.delete(trainX, np.s_[10:len(trainX)], axis=1)  
    
    trainX, trainY, validX, validY  = multiT(trainX, trainY, validX, validY) 
    test = testX.T, testY.T, testy
    Xt, Yt, yt = test

    val       = 1e-5
    k         = np.size(trainY, 0)
    d         = np.size(trainX, 0)
    n         = np.size(trainX, 1)
    m         = 10

    n_batches = 450 
    n_b_size  = 100 
    lmbda     = 0.0044668359215096305
    l_mod     = 0.0002
    
    cycles    = 4
    eta_min   = 1e-5
    eta_max   = 1e-1
    n_s       = int(5 * int(n / n_b_size))
    t         = cycles*2*n_s
    etas      = eta_min, eta_max, n_s, t, cycles

    networkDim = [d, 50, 30, 20, 20, 10, 10, 10, 10, k]
    """ Use only when testing sensitivity to initialisation"""
#    weightsAndBias = createParamsSensitive(networkDim, 1e-1) 
    weightsAndBias = createParams(networkDim)
    Gamma, Beta =           createYB(networkDim)
    GG                      = trainX, trainY, trainy, validX, validY, vaildy

    batchNorm = True
    printStuff = True

    
    """ For testing gradients, dont forget to reduce dimensionality on row 591"""
#    testGradients(trainX, trainY, trainy, 1000, weightsAndBias, Gamma, Beta, lmbda, batchNorm)

    """ Mini batchGD"""
 #   params, gamma, beta, mu, v, finalAcc   = miniBatchGD(GG, n_batches, etas, weightsAndBias, Gamma, Beta, lmbda, batchNorm, printStuff)

    """ For finding best lambda: """
#    bestParams(GG, n_batches, etas, weightsAndBias, Gamma, Beta, lmbda, test, batchNorm, 20) # for finding best lambda 

    ensamble = True
    if(ensamble):

        netA = [d, 2000, 1000, 2000, k]
        netB = [d, 2000, 1000, 2000, k]
        netC = [d, 500, 1000, 1000, 500, k]
        netD = [d, 1000, 4000, 1000, k]
        netE = [d, 3000, 3000, k]

        weightsAndBiasA     = createParams(netA)
        weightsAndBiasB     = createParams(netB)
        weightsAndBiasC     = createParams(netC)
        weightsAndBiasD     = createParams(netD)
        weightsAndBiasE     = createParams(netE)

        AGamma, ABeta =           createYB(netA)
        BGamma, BBeta =           createYB(netB)
        CGamma, CBeta =           createYB(netC)
        DGamma, DBeta =           createYB(netD)
        EGamma, EBeta =           createYB(netE)

        paramsC, gammaC, betaC, muC, vC, finalAccC   = miniBatchGD(GG, n_batches, etas, weightsAndBiasC, CGamma, CBeta, lmbda, batchNorm, printStuff)
        paramsA, gammaA, betaA, muA, vA, finalAccA   = miniBatchGD(GG, n_batches, etas, weightsAndBiasA, AGamma, ABeta, lmbda, batchNorm, printStuff)
        paramsB, gammaB, betaB, muB, vB, finalAccB   = miniBatchGD(GG, n_batches, etas, weightsAndBiasB, BGamma, BBeta, 0.03, batchNorm, printStuff)
        paramsD, gammaD, betaD, muD, vD, finalAccD   = miniBatchGD(GG, n_batches, etas, weightsAndBiasD, DGamma, DBeta, lmbda+0.0002, batchNorm, printStuff)
        paramsE, gammaE, betaE, muE, vE, finalAccE   = miniBatchGD(GG, n_batches, etas, weightsAndBiasE, EGamma, EBeta, lmbda-0.0002, batchNorm, printStuff)

        print("######## i is: "+ str(i) +" ##########")
        print("individual test accuracy: ")
        print( str( ComputeAcc(Xt, yt, paramsA, gammaA, betaA, muA, vA, batchNorm)))
        print( str( ComputeAcc(Xt, yt, paramsB, gammaB, betaB, muB, vB, batchNorm)))
        print( str( ComputeAcc(Xt, yt, paramsC, gammaC, betaC, muC, vC, batchNorm)))
        print( str( ComputeAcc(Xt, yt, paramsD, gammaD, betaD, muD, vD, batchNorm)))
        print( str( ComputeAcc(Xt, yt, paramsE, gammaE, betaE, muE, vE, batchNorm)))

        print("---------------------------")
        netsA = paramsA, gammaA, betaA, muA, vA
        netsB = paramsB, gammaB, betaB, muB, vB
        netsC = paramsC, gammaC, betaC, muC, vC
        netsD = paramsD, gammaD, betaD, muD, vD
        netsE = paramsE, gammaE, betaE, muE, vE

        wholeTeam           = netsA, netsB, netsC, netsD, netsE
        outer               = netsA, netsE
        middleAndOuter      = netsA, netsC, netsE
        vampire             = netsB, netsD
        middleThree         = netsB, netsC, netsD
        allButMiddle        = netsA, netsB, netsD, netsE

        acc_wholeTeam       = ComputeEnsambleAcc(Xt, yt, wholeTeam, batchNorm)
        acc_outer           = ComputeEnsambleAcc(Xt, yt, outer, batchNorm)
        acc_middleAndOuter  = ComputeEnsambleAcc(Xt, yt, middleAndOuter, batchNorm)
        acc_vampire         = ComputeEnsambleAcc(Xt, yt, vampire, batchNorm)
        acc_middleThree     = ComputeEnsambleAcc(Xt, yt, middleThree, batchNorm)
        acc_allButMiddle    = ComputeEnsambleAcc(Xt, yt, allButMiddle, batchNorm)
   
        print("wholeTeam aensemble accuracy:     " + str(acc_wholeTeam))
        print("outer ensemble accuracy:          " + str(acc_outer))
        print("middleAndOuter ensemble accuracy: " + str(acc_middleAndOuter))
        print("vampire ensemble accuracy:        " + str(acc_vampire))
        print("middleThree ensemble accuracy:    " + str(acc_middleThree))
        print("allButMiddle ensemble accuracy:   " + str(acc_allButMiddle))
        print("##################################")
    