import unittest

from EM import *
import numpy as np
from itertools import product
import random

LOG_1 = 0

def randomSequence(length, max_char):
    return [ random.randint(0, max_char-1) for _ in range(length)]

class TestSafeLogAdd(unittest.TestCase):
    def test_vs_easy_log(self):
        x = np.array([5,2,3,4,5,6])
        self.assertAlmostEqual(safeLogAdd(x), np.log(np.sum(np.exp(x))))

    def test_log_vs_real(self):
        x = np.array([5,2,3,4,5,6])
        self.assertAlmostEqual(np.exp(safeLogAdd(np.log(x))), np.sum(x))

    def test_log_vs_real_zero(self):
        x = np.array([0,2,3,4,5,6])
        self.assertAlmostEqual(np.exp(safeLogAdd(np.log(x))), np.sum(x))

    def test_log_vs_real_large(self):
        x = np.random.rand(100000)*100
        self.assertAlmostEqual(np.exp(safeLogAdd(np.log(x))), np.sum(x))

class TestModelCreate(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)
    
    def test_pi(self):
        self.assertEqual(self.model.pi.shape, (self.hidden,))
        self.assertAlmostEqual(safeLogAdd(self.model.pi), LOG_1)

    def test_m(self):
        self.assertEqual(self.model.m.shape, (self.hidden, self.hidden))
        
        for i in range(self.hidden):
            self.assertAlmostEqual(safeLogAdd(self.model.m[i]), LOG_1)

    def test_e(self):
        self.assertEqual(self.model.e.shape, (self.hidden, self.observe))

        for i in range(self.hidden):
            self.assertAlmostEqual(safeLogAdd(self.model.e[i]), LOG_1)

    def test_meta(self):
        self.assertEqual(self.model.hidden, self.hidden)
        self.assertEqual(self.model.observeable, self.observe)

class TestCalcAlpha(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)
    
    def test_base_case(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, [0,1,2])
        
        self.assertEqual(alpha[0,0], self.model.pi[0]+self.model.e[0, obsStates[0]])
        self.assertEqual(alpha[0,1], self.model.pi[1]+self.model.e[1, obsStates[0]])

    def test_base_case_sum_1(self):
        initial_alphas = []
        for i in range(self.observe): # alpha[0] = initial*obs[0], sum across all obs
            alpha = calculateAlpha(self.model, [i])
            initial_alphas.append(safeLogAdd(alpha[0]))

        self.assertAlmostEqual(safeLogAdd(np.array(initial_alphas)), LOG_1)

    def test_inductive_case(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, [0,1,2])
        
        j = 0
        expected = safeLogAdd(alpha[0]+self.model.m[:,j])+self.model.e[j, obsStates[1]]
        self.assertEqual(alpha[1,0], expected)
        
    def test_inductive_case_sum_1(self):
        
        all_sequences = list(product(range(0, self.observe), repeat=5))
        all_alphas = []

        for s in all_sequences:
            alpha = calculateAlpha(self.model, s)
            all_alphas.append(safeLogAdd(alpha[-1]))

        self.assertAlmostEqual(safeLogAdd(np.array(all_alphas)), LOG_1)

    def test_full(self):
        O = randomSequence(30, self.observe)
        alpha = calculateAlpha(self.model, O)
        pseudo_alpha = np.ones((len(O), self.hidden))
        
        for i in range(self.hidden):
            pseudo_alpha[0, i] = self.model.pi[i] + self.model.e[i, O[0]]

        for t in range(0, len(O)-1):
            for j in range(self.hidden):
                parts = []
                for i in range(self.hidden):
                    parts.append(pseudo_alpha[t, i]+self.model.m[i,j])
                
                pseudo_alpha[t+1,j] = self.model.e[j, O[t+1]] + safeLogAdd(np.array(parts))

        self.assertTrue(np.allclose(alpha, pseudo_alpha))


class TestCalcBeta(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)

    def test_base_case(self):
        obsStates = [0,1,2]
        beta = calculateBeta(self.model, [0,1,2])

        self.assertTrue(np.array_equal(beta[-1], np.log(np.ones(self.hidden))))

    
    def test_inductive_case(self):
        obsStates = [0,1,2]
        beta = calculateBeta(self.model, [0,1,2])
        
        i = 0
        expected = safeLogAdd(self.model.m[i]+self.model.e[:,obsStates[-1]]+beta[-1])
        self.assertEqual(beta[-2, i], expected)

    
    def test_inductive_case_sum_1(self):
        obsStates = [0,1,2]
        beta = calculateBeta(self.model, [0,1,2])
        
        betas = []
        # sum over all paths for a particular state
        for i in range(self.observe):
            for j in range(self.observe):
                beta = calculateBeta(self.model, [0, i, j])
                betas.append(beta[0,0])

        self.assertAlmostEqual(safeLogAdd(np.array(betas)), LOG_1)

    def test_full(self):
        O = randomSequence(30, self.observe)
        beta = calculateBeta(self.model, O)
        
        pseudo_beta = np.ones((len(O), self.hidden))
        
        for i in range(self.hidden):
            pseudo_beta[-1, i] = LOG_1

        for t in reversed(range(len(O)-1)):
            for i in range(self.hidden):
                parts = []
                for j in range(self.hidden):
                    parts.append(self.model.m[i, j]+self.model.e[j,O[t+1]]+pseudo_beta[t+1, j])
                
                pseudo_beta[t,i] = safeLogAdd(np.array(parts))

        self.assertTrue(np.allclose(beta, pseudo_beta))

class TestCalcGamma(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)

    def test_single_value(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)

        # Gamma values are normalized over each timestep
        # To avoid normalising in test check ratio between expected and calc vals
        unnorm_v1 = alpha[0,0] + beta[0,0]
        unnorm_v2 = alpha[0,1] + beta[0,1]

        r1 = unnorm_v1-unnorm_v2
        r2 = gamma[0,0]-gamma[0,1]
        
        self.assertAlmostEqual(r1, r2)
    
    def test_t_sum_1(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)

        for i in range(self.observe):
            self.assertAlmostEqual(safeLogAdd(gamma[i]), LOG_1)

    def test_full(self):
        O = randomSequence(30, self.observe)
        alpha = calculateAlpha(self.model, O)
        beta = calculateBeta(self.model, O)

        gamma = calculateGamma(self.model, O, alpha, beta)

        pseudo_gamma = np.ones((len(O), self.hidden))
        
        for t in range(len(O)):
            for i in range(self.hidden):
                parts = []
                for j in range(self.hidden):
                    parts.append(alpha[t, j]+beta[t, j])
                denom = safeLogAdd(np.array(parts))
            
                pseudo_gamma[t, i] = alpha[t,i]+beta[t,i] - denom 

        self.assertTrue(np.allclose(gamma, pseudo_gamma))

        
       


class TestCalcXi(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)

    def test_single_value(self):
        obsStates = [0,1,2,0,0]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        i_1 = 1
        j_1 = 0
        i_2 = 2
        j_2 = 3
        t = 3
        unnorm_v1 = alpha[t, i_1]+ self.model.m[i_1, j_1]+self.model.e[j_1, obsStates[t+1]]+beta[t+1, j_1]
        unnorm_v2 = alpha[t, i_2]+ self.model.m[i_2, j_2]+self.model.e[j_2, obsStates[t+1]]+beta[t+1, j_2]

        r1 = unnorm_v1-unnorm_v2
        r2 = xi[t, i_1, j_1]-xi[t, i_2, j_2]
        
        self.assertAlmostEqual(r1, r2)

    def test_t_sum_1(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        for t in range(len(obsStates)-1):
            self.assertAlmostEqual(safeLogAdd(xi[t].reshape(-1)), LOG_1)

    def test_j_sum_gamma(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        for t in range(len(obsStates)-1):
            for i in range(self.model.hidden):
                self.assertAlmostEqual(safeLogAdd(xi[t, i]), gamma[t,i])

    def test_full(self):
        O = randomSequence(30, self.observe)
        alpha = calculateAlpha(self.model, O)
        beta = calculateBeta(self.model, O)

        gamma = calculateGamma(self.model, O, alpha, beta)

        xi = calculateXi(self.model, O, alpha, beta, gamma)

        pseudo_xi = np.ones((len(O)-1, self.hidden, self.hidden))
        
        for t in range(len(O)-1):
            for i in range(self.hidden):
                for j in range(self.hidden):

                    parts = []
                    for x in range(self.hidden):
                        for y in range(self.hidden):
                            parts.append(alpha[t, x]+self.model.m[x,y]+self.model.e[y, O[t+1]]+beta[t+1, y])

                    denom = safeLogAdd(np.array(parts))

                    pseudo_xi[t, i, j] = alpha[t,i]+self.model.m[i, j]+self.model.e[j, O[t+1]]+beta[t+1, j] - denom 

        self.assertTrue(np.allclose(xi, pseudo_xi))


class TestIterModel(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)

    def test_pi_sum_1(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        new_model = iterateModel(self.model, obsStates, gamma, xi)
        
        self.assertAlmostEqual(safeLogAdd(new_model.pi), LOG_1)

    def test_m_sum_1(self):
        obsStates = [0,1,2,1,1]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        new_model = iterateModel(self.model, obsStates, gamma, xi)
        for i in range(self.hidden):
            self.assertAlmostEqual(safeLogAdd(new_model.m[i]), LOG_1)

    def test_e_sum_1(self):
        obsStates = [0,1,1,1]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        new_model = iterateModel(self.model, obsStates, gamma, xi)

        for i in range(self.hidden):
            self.assertAlmostEqual(safeLogAdd(new_model.e[i]), LOG_1)

    def createNewModel(self, starting_model, O):
        alpha = calculateAlpha(starting_model, O)
        beta = calculateBeta(starting_model, O)
        gamma = calculateGamma(starting_model, O, alpha, beta)
        xi = calculateXi(starting_model, O, alpha, beta, gamma)
        new_model = iterateModel(starting_model, O, gamma, xi)

        return alpha, beta, gamma, xi, new_model
    
    def test_pi_full(self):
        O = randomSequence(30, self.observe)
        alpha, beta, gamma, xi, new_model = self.createNewModel(self.model, O)
        
        pseudo_pi = np.ones((self.hidden))

        for i in range(self.hidden):
            pseudo_pi[i] = gamma[0, i]

        self.assertTrue(np.allclose(new_model.pi, pseudo_pi))

class TestMainFunction(unittest.TestCase):
    hidden = 6
    observe = 5
    seq = [0,0,0,1,1,2,3,1,1,1,1,0,3,3,3,3,3,3,3,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,1,1,1,1,1,2,1,2,1,2,1,2,1]
    model = estimateModel(seq, hidden, observe)

    def test_pi_sum_1(self):
        pi = self.model[0]
        self.assertAlmostEqual(np.sum(pi), 1)

    def test_m_sum_1(self):
       
        m = self.model[1]
        for i in range(self.hidden):
            self.assertAlmostEqual(np.sum(m[i]), 1)

    def test_e_sum_1(self):
        e = self.model[2]
        for i in range(self.hidden):
            self.assertAlmostEqual(np.sum(e[i]), 1)


if __name__ == '__main__':
    unittest.main()