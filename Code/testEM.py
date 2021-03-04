import unittest

from EM import *
import numpy as np


class TestModelCreate(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)
    
    def test_pi(self):
        self.assertEqual(self.model.pi.shape, (self.hidden,))
        self.assertEqual(np.sum(self.model.pi), 1)

    def test_m(self):
        self.assertEqual(self.model.m.shape, (self.hidden, self.hidden))
        self.assertTrue(np.array_equal(np.sum(self.model.m, axis=1), np.ones(self.hidden)))

    def test_e(self):
        self.assertEqual(self.model.e.shape, (self.hidden, self.observe))
        self.assertTrue(np.array_equal(np.sum(self.model.e, axis=1), np.ones(self.hidden)))

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
        
        self.assertEqual(alpha[0,0], self.model.pi[0]*self.model.e[0, obsStates[0]])
        self.assertEqual(alpha[0,1], self.model.pi[1]*self.model.e[1, obsStates[0]])

    def test_base_case_sum_1(self):
        summed = 0
        for i in range(self.observe):
            alpha = calculateAlpha(self.model, [i])
            summed += np.sum(alpha[0])

        self.assertEqual(summed, 1)

    def test_inductive_case(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, [0,1,2])
        
        j = 0
        expected = np.sum(alpha[0]*self.model.m[:,j])*self.model.e[j, obsStates[1]]
        self.assertEqual(alpha[1,0], expected)
        
    def test_inductive_case_sum_1(self):
        
        summed = 0
        for i in range(self.observe):
            for j in range(self.observe):
                alpha = calculateAlpha(self.model, [i, j])
                summed += np.sum(alpha[1])

        self.assertAlmostEqual(summed, 1.0)


class TestCalcBeta(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)

    def test_base_case(self):
        obsStates = [0,1,2]
        beta = calculateBeta(self.model, [0,1,2])

        self.assertTrue(np.array_equal(beta[-1], np.ones(self.hidden)))

    
    def test_inductive_case(self):
        obsStates = [0,1,2]
        beta = calculateBeta(self.model, [0,1,2])
        
        i = 0
        expected = np.sum(self.model.m[i]*self.model.e[:,obsStates[-1]]*beta[-1])
        self.assertEqual(beta[-2, i], expected)

    
    def test_inductive_case_sum_1(self):
        obsStates = [0,1,2]
        beta = calculateBeta(self.model, [0,1,2])
        
        i = 0
        expected = np.sum(self.model.m[i]*self.model.e[:,obsStates[-1]]*beta[-1])
        summed = 0

        # sum over all paths for a particular state
        for i in range(self.observe):
            for j in range(self.observe):
                beta = calculateBeta(self.model, [0, i, j])
                summed += beta[0,0]

        self.assertAlmostEqual(summed, 1.0)

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
        unnorm_v1 = alpha[0,0] * beta[0,0]
        unnorm_v2 = alpha[0,1] * beta[0,1]

        r1 = unnorm_v1/unnorm_v2
        r2 = gamma[0,0]/gamma[0,1]
        
        self.assertEqual(r1, r2)
    
    def test_t_sum_1(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)

        for i in range(self.observe):
            self.assertAlmostEqual(np.sum(gamma[i]), 1)


class TestCalcXi(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)

    def test_single_value(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        i_1 = 0
        j_1 = 0
        i_2 = 0
        j_2 = 0
        t = 1
        unnorm_v1 = alpha[t, i_1]* self.model.m[i_1, j_1]*self.model.e[j_1, obsStates[t+1]]*beta[t+1, j_1]
        unnorm_v2 = alpha[t, i_2]* self.model.m[i_2, j_2]*self.model.e[j_2, obsStates[t+1]]*beta[t+1, j_2]

        r1 = unnorm_v1/unnorm_v2
        r2 = xi[t, i_1, j_1]/xi[t, i_2, j_2]
        
        self.assertEqual(r1, r2)

    def test_t_sum_1(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        for t in range(len(obsStates)-1):
            self.assertAlmostEqual(np.sum(xi[t]), 1.0)

    def test_j_sum_gamma(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        for t in range(len(obsStates)-1):
            for i in range(self.model.hidden):
                self.assertAlmostEqual(np.sum(xi[t, i]), gamma[t,i])


if __name__ == '__main__':
    unittest.main()