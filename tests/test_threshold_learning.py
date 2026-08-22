import unittest

import numpy as np

from covid_agpt.threshold_learning import (
    deterministic_1d_kmeans,
    sklearn_1d_kmeans,
)


class ThresholdLearningTests(unittest.TestCase):
    def setUp(self):
        self.values = np.array([0.01, 0.02, 0.03, 0.5, 0.55, 0.6, 2.0, 2.1, 2.2])

    def test_deterministic_centers_are_sorted(self):
        result = deterministic_1d_kmeans(self.values, k=3)
        self.assertTrue(np.all(np.diff(result.centers) > 0))
        self.assertEqual(len(result.labels), len(self.values))

    def test_sklearn_is_reproducible(self):
        first = sklearn_1d_kmeans(self.values, k=3)
        second = sklearn_1d_kmeans(self.values, k=3)
        np.testing.assert_allclose(first.centers, second.centers)
        np.testing.assert_allclose(first.cutoffs, second.cutoffs)


if __name__ == "__main__":
    unittest.main()
