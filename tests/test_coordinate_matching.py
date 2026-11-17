import unittest
import numpy as np
from coordinate_matching import contains_coordinate

class CoordinateMatchingTests(unittest.TestCase):
    def test_scalar_membership_is_not_a_coordinate(self):
        ideal = [0.1, 0.2, 0.3]
        defects = [[0.1, 0.8, 0.7], [0.9, 0.2, 0.3]]
        self.assertTrue(np.isin(ideal, defects, assume_unique=True).all())
        self.assertFalse(contains_coordinate(ideal, defects))
    def test_present_coordinate(self):
        self.assertTrue(contains_coordinate([0.1, 0.2, 0.3], [[0.1, 0.2, 0.3]]))
    def test_empty(self):
        self.assertFalse(contains_coordinate([0.1, 0.2, 0.3], []))
    def test_coordinate_order_matters(self):
        self.assertFalse(contains_coordinate([0.1, 0.2, 0.3], [[0.2, 0.1, 0.3]]))
    def test_five_decimal_convention(self):
        self.assertTrue(contains_coordinate([0.100001, 0.2, 0.3], [[0.1, 0.2, 0.3]]))
if __name__ == '__main__': unittest.main()
