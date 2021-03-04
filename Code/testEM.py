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

        for i in range(self.observe):
            for j in range(self.observe):
                beta = calculateBeta(self.model, [0, i, j])
                summed += beta[0,0]

        self.assertAlmostEqual(summed, 1.0)

if __name__ == '__main__':
    unittest.main()