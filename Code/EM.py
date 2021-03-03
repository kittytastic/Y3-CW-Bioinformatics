import argparse
import os
import numpy as np


class Model():
    def __init__(self, pi, m, e):
        self.pi = pi
        self.m = m
        self.e = e

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

   
