import unittest

from EM import *
import numpy as np

LOG_1 = 0

class TestModelCreate(unittest.TestCase):
    hidden = 5
    observe = 3
    model = createInitialModel(hidden, observe)
    
    def test_pi(self):
        self.assertEqual(self.model.pi.shape, (self.hidden,))
        #self.assertAlmostEqual(np.log(np.sum(np.exp(self.model.pi))), 0)
        self.assertAlmostEqual(logAddExp(self.model.pi), LOG_1)

    def test_m(self):
        self.assertEqual(self.model.m.shape, (self.hidden, self.hidden))
        
        for i in range(self.hidden):
            self.assertAlmostEqual(logAddExp(self.model.m[i]), LOG_1)

    def test_e(self):
        self.assertEqual(self.model.e.shape, (self.hidden, self.observe))

        for i in range(self.hidden):
            self.assertAlmostEqual(logAddExp(self.model.e[i]), LOG_1)

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
            initial_alphas.append(logAddExp(alpha[0]))

        self.assertAlmostEqual(logAddExp(initial_alphas), LOG_1)

    def test_inductive_case(self):
        obsStates = [0,1,2]
        alpha = calculateAlpha(self.model, [0,1,2])
        
        j = 0
        expected = logAddExp(alpha[0]+self.model.m[:,j])+self.model.e[j, obsStates[1]]
        self.assertEqual(alpha[1,0], expected)
        
    def test_inductive_case_sum_1(self):
        
        all_alphas = []
        for i in range(self.observe):
            for j in range(self.observe):
                alpha = calculateAlpha(self.model, [i, j])
                all_alphas.append(logAddExp(alpha[1]))

        self.assertAlmostEqual(logAddExp(all_alphas), LOG_1)


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
        
        self.assertAlmostEqual(r1, r2)
    
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
        unnorm_v1 = alpha[t, i_1]* self.model.m[i_1, j_1]*self.model.e[j_1, obsStates[t+1]]*beta[t+1, j_1]
        unnorm_v2 = alpha[t, i_2]* self.model.m[i_2, j_2]*self.model.e[j_2, obsStates[t+1]]*beta[t+1, j_2]

        r1 = unnorm_v1/unnorm_v2
        r2 = xi[t, i_1, j_1]/xi[t, i_2, j_2]
        
        self.assertAlmostEqual(r1, r2)

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
        
        self.assertAlmostEqual(np.sum(new_model.pi), 1.0)

    def test_m_sum_1(self):
        obsStates = [0,1,2,1,1]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        new_model = iterateModel(self.model, obsStates, gamma, xi)
        for i in range(self.hidden):
            self.assertAlmostEqual(np.sum(new_model.m[i]), 1.0)

    def test_e_sum_1(self):
        obsStates = [0,1,1,1]
        alpha = calculateAlpha(self.model, obsStates)
        beta = calculateBeta(self.model, obsStates)

        gamma = calculateGamma(self.model, obsStates, alpha, beta)
        xi = calculateXi(self.model, obsStates, alpha, beta, gamma)

        new_model = iterateModel(self.model, obsStates, gamma, xi)

        for i in range(self.hidden):
            self.assertAlmostEqual(np.sum(new_model.e[i]), 1.0)



if __name__ == '__main__':
    unittest.main()