import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import math


'''
Main function:

Runs the Baum-Welch Algorithm until model reaches a local optimum

Arg:
    observedSeq: The observed sequence as a List of integers
    numHiddenStates: The number of hidden states in the model as an integer
    numObserveableStates: The size of the alphabet as an integer
    verbose (optional): default verbose=False

Returns (as tuple):
    [0] Initial Probabilites: numpy array (1D) 
    [1] Transition Probabilites: numpy array (2D) indexed as [from state si, to state sj]
    [2] Emission Probabilities: numpy array (2D) indexed as [from state si, to symbol l]
    [3] p(observation|model) for all iterations of the model: List
'''

def estimateModel(observedSeq, numHiddenStates, numObserveableStates, verbose=False):

    model = createInitialModel(numHiddenStates, numObserveableStates)
    if verbose:
        print("P initial: %.2f"%probOfObservations(model, observedSeq))
    

    iterProb = []
    lastP = 0.0
    iterProb.append(lastP)
    i = 0
    converged = False
    while not converged:
        i += 1
        alpha = calculateAlpha(model, observedSeq)
        beta = calculateBeta(model, observedSeq)
        gamma = calculateGamma(model, observedSeq, alpha, beta)
        xi = calculateXi(model, observedSeq, alpha, beta, gamma)
        model = iterateModel(model, observedSeq, gamma, xi)
        
        newP = probOfObservations(model, observedSeq)
        iterProb.append(newP)

        if verbose:
            print("P %d: %.2e"%( i, newP))

        if math.isclose(lastP, newP):
            converged = True
        
        lastP = newP

    return model.pi, model.m, model.e, iterProb

class Model():
    def __init__(self, pi, m, e):
        self.pi = pi # [hidden]
        self.m = m # [from, to]
        self.e = e # [hidden, observed]
        self.hidden, self.observeable = e.shape 

    def __str__(self):
        outS = "--- pi ---\n"
        outS += str(self.pi)
        outS += "\n--- m ---\n"
        outS += str(self.m)
        outS += "\n--- e ---\n"
        outS += str(self.e)

        return outS

def logAddExp(*args):
    tmp = np.array(args)
    return np.log(np.sum(np.exp(tmp)))

def safeLogAdd(x):
    assert(len(x.shape)==1)
    if len(x)==0:
        return np.NINF
    special = x.max()
    i_special = x.argmax()
    remaining = x[np.arange(len(x))!=i_special]
    y = remaining - special
    return special + np.log1p(np.sum(np.exp(y)))


def createInitialModel(hidden_states, observeable_states):
    #pi = np.ones(hidden_states)
    #pi /= hidden_states

    #m = np.ones((hidden_states, hidden_states))
    #for i in range(hidden_states):
    #    m[i] /= hidden_states

    #e = np.ones((hidden_states, observeable_states))
    #for i in range(hidden_states):
    #    e[i] /= observeable_states

    pi = np.random.rand(hidden_states)
    pi /= pi.sum()
    pi = np.log(pi)

    m = np.random.rand(hidden_states, hidden_states)
    m /= np.expand_dims(m.sum(axis=1), axis=1)
    m = np.log(m)

    e = np.random.rand(hidden_states, observeable_states)
    e /= np.expand_dims(e.sum(axis=1), axis=1)
    e = np.log(e)

    return Model(pi, m, e)

def calculateAlpha(model, observedSeq):
    alpha = np.zeros((len(observedSeq), model.hidden))

    # Base case: alpha_1(i) = pi_i * e_i(O_1)
    alpha[0] = model.pi + model.e[:, observedSeq[0]]
    

    # Inductive case: alpha_{t+1}(j) = [\sigma_{i=1}^N alpha_t(i) m_ij ] * e_i(O_1)
    # i.e. Prob state j at t+1 = sum (prob state i at t * transition i->j)  * prob emit obs t+1
    for t in range(1, len(observedSeq), 1):
        for j in range(model.hidden):
            m__j = model.m[:,j]
            a_t = alpha[t-1]
            alpha[t, j] = safeLogAdd(a_t+m__j)+model.e[j, observedSeq[t]]
        

    return alpha

def calculateBeta(model, observedSeq):
    beta = np.ones((len(observedSeq), model.hidden))

    # Base case
    beta[-1] = np.zeros(model.hidden) # log(1) = 0

    # Inductive case
    i = 0

    for t in reversed(range(len(observedSeq)-1)):
        for i in range(model.hidden):
            beta[t, i] = safeLogAdd(model.m[i]+model.e[:,observedSeq[t+1]]+beta[t+1])

    return beta

def calculateGamma(model, observedSeq, alpha, beta):
    gamma = np.ones((len(observedSeq), model.hidden))

    gamma = alpha + beta

    for t in range(len(observedSeq)):
        gamma[t] -= safeLogAdd(alpha[t]+beta[t])

    return gamma

def calculateXi(model, observedSeq, alpha, beta, gamma):
    xi = np.zeros((len(observedSeq)-1, model.hidden, model.hidden))

    for t in range(len(observedSeq)-1):
        e_obs = model.e[:,observedSeq[t+1]] 
        beta_t = beta[t+1]
        alpha_t = alpha[t]

        j_row = e_obs + beta_t

        ts = np.add.outer(alpha_t, j_row) + model.m
        xi[t] = ts - safeLogAdd(ts.reshape(-1))

    return xi


def iterateModel(model, observedSeq, gamma, xi):
    pi = gamma[1]

    m = np.zeros((model.hidden, model.hidden))
    for i in range(model.hidden):
        gammas = gamma[:-1, i]
        assert(len(gammas) == len(gamma)-1)
        denom = safeLogAdd(gammas)
        for j in range(model.hidden):
            xis = xi[:, i, j]
            m[i,j] = safeLogAdd(xis)-denom

    e = np.ones((model.hidden, model.observeable))
    for i in range(model.hidden):
        denom = safeLogAdd(gamma[:,i])
        for k in range(model.observeable):
            gamma_ik_mask = np.where(np.array(observedSeq)==k, True, False)
            gamma_ik = gamma[:,i]
            gamma_ik = gamma_ik[gamma_ik_mask]
            e[i,k] = safeLogAdd(gamma_ik) - denom
        
            
    
    return Model(pi, m, e)

def probOfObservations(model, observedSeq):
    alpha = calculateAlpha(model, observedSeq)
    p = np.exp(safeLogAdd(alpha[-1]))
    return p

if __name__ =="__main__":
    #observedSeq = [0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1]
    observedSeq = [0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,2,0,0,0,1,1,2,1,1,1,1,1,1,1,1,1,0,0,0,1,1]
    #observedSeq = [0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3]
    hidden = 50
    observeable = 5

    for i in range(5):
        _, _, _, probs = estimateModel(observedSeq, hidden, observeable, verbose=True) 
        plt.plot(probs, color='C%d'%i)
    plt.ylabel('$p(x|\lambda)$')
    plt.yscale('log')

    plt.xlabel('Model iteration')
    plt.savefig("temp.png")

   
