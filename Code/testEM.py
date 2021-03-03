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


if __name__ == '__main__':
    unittest.main()