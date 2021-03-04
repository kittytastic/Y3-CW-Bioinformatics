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

    #print(model.e)
    #print(model.e[:, observedSeq[0]])
    #print(model.pi)
    #print(model.e[:, observedSeq[0]])
    alpha[0] = model.pi * model.e[:, observedSeq[0]]
    
    #model.m[0, 0] = 0.1 
    #model.m[0,-1] = 0.3
    #print(model.m) 
    #print(model.m[0])

    m__j = model.m[:,0]
    a_t = alpha[0]
    #print(m__j)
    #print(a_t)

    #print(model.e[0, 0])
    #print(a_t*m__j)
    #print(np.sum(a_t*m__j))
    #print()


    for t in range(1, len(observedSeq), 1):
        for j in range(model.hidden):
            m__j = model.m[:,j]
            a_t = alpha[t-1]
            alpha[t, j] = np.sum(a_t*m__j)*model.e[j, observedSeq[t]]
        

    return alpha

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

   
