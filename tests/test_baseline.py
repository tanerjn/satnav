import unittest
import numpy as np
from satnav.dynamics import circular_state, propagate, invariants
from satnav.elm_model import ELM

class BaselineTests(unittest.TestCase):
    def test_orbit_invariants(self):
        t, x = propagate(circular_state(), 5400, 10)
        e, h = invariants(x)
        self.assertEqual(x.shape, (541,6))
        self.assertLess(np.max(np.abs(e/e[0]-1)), 1e-7)
        self.assertLess(np.max(np.abs(h/h[0]-1)), 1e-7)

    def test_elm_regression(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(-1,1,(100,3))
        Y = np.column_stack([np.tanh(X[:,0]), X[:,1]])
        a = ELM(hidden=40,seed=42).fit(X,Y)
        self.assertEqual(a.predict(X[:5]).shape, (5,2))
        self.assertTrue(np.isfinite(a.predict(X)).all())

if __name__ == "__main__":
    unittest.main()
