import argparse
import os
import numpy as np


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

def createInitialModel(hidden_states, observeable_states):
    pi = np.ones(hidden_states)
    pi /= hidden_states

    m = np.ones((hidden_states, hidden_states))
    for i in range(hidden_states):
        m[i] /= hidden_states

    e = np.ones((hidden_states, observeable_states))
    for i in range(hidden_states):
        e[i] /= observeable_states

    return Model(pi, m, e)

def calculateAlpha(model, observedSeq):
    alpha = np.zeros((len(observedSeq), model.hidden))

    # Base case: alpha_1(i) = pi_i * e_i(O_1)
    alpha[0] = model.pi * model.e[:, observedSeq[0]]
    

    # Inductive case: alpha_{t+1}(j) = [\sigma_{i=1}^N alpha_t(i) m_ij ] * e_i(O_1)
    # i.e. Prob state j at t+1 = sum (prob state i at t * transition i->j)  * prob emit obs t+1
    for t in range(1, len(observedSeq), 1):
        for j in range(model.hidden):
            m__j = model.m[:,j]
            a_t = alpha[t-1]
            alpha[t, j] = np.sum(a_t*m__j)*model.e[j, observedSeq[t]]
        

    return alpha

def calculateBeta(model, observedSeq):
    beta = np.zeros((len(observedSeq), model.hidden))

    # Base case
    beta[-1] = np.ones(model.hidden)

    # Inductive case
    i = 0
    seq_i = -1
    ans = np.sum(model.m[i]*model.e[:,observedSeq[-1]]*beta[-1])

    for t in reversed(range(len(observedSeq)-1)):
        for i in range(model.hidden):
            beta[t, i] = np.sum(model.m[i]*model.e[:,observedSeq[t+1]]*beta[t+1])

    return beta

def calculateGamma(model, observedSeq, alpha, beta):
    gamma = np.zeros((len(observedSeq), model.hidden))

    gamma = alpha * beta

    for t in range(len(observedSeq)):
        gamma[t] /= np.sum(gamma[t])

    return gamma

def calculateXi(model, observedSeq, alpha, beta, gamma):
    xi = np.zeros((len(observedSeq)-1, model.hidden, model.hidden))

    for t in range(len(observedSeq)-1):
        e_obs = model.e[:,observedSeq[t+1]] 
        beta_t = beta[t+1]
        alpha_t = alpha[t]

        j_row = e_obs * beta_t

        ts = np.outer(alpha_t, j_row) * model.m
        xi[t] = ts / np.sum(ts)

    return xi


def iterateModel(model, observedSeq, gamma, xi):
    pi = gamma[1]

    # only sum over 0 -> T-1, easy for xi, fiddle for gamma
    m = np.sum(xi, axis=0)/(np.sum(gamma, axis=0)-gamma[-1])

    e = np.ones((model.hidden, model.observeable))
    for i in range(model.hidden):
        for k in range(model.observeable):
            gamma_ik_mask = np.where(np.array(observedSeq)==k, True, False)
            gamma_ik = np.sum(gamma[:,i], where=gamma_ik_mask)
            e[i,k]=gamma_ik
        
        e[i] /= np.sum(e[i])
            
    
    return Model(pi, m, e)

def assertFileExists(path):
    if not os.path.isfile(path):
        print("ERROR: %s doesn't exist."%path) 
        exit()

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='EM')
    #parser.add_argument("file_path")
    parser.add_argument("-k", dest="num_states", help="Number of hidden states")
    parser.add_argument("-s", dest="num_alpabet", help="Number of symbols in alphabet")
    #parser.add_argument("-k", dest="num_states", help="Number of hidden states", action='store_true')
    #parser.add_argument("--type", dest="testType", help="Select a test type from:",default="live" )
    
    args = parser.parse_args()
    #os.environ['TRIAL'] = "T" if args.trial else "F"
    
    num_hidden_states = int(args.num_states)
    num_alphabet_sym = int(args.num_alpabet)

    model = createInitialModel(num_hidden_states, num_alphabet_sym)

    print(model)
    print(np.sum(model.e, axis=1))

   
