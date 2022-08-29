import unittest
from unittest.mock import patch


from src.utils.util_main_ref import *


class test_utils_main(unittest.TestCase):
    def test_load_cfg(self):
        testargs = ['main', 'tests/dummy_configs/config_test.yaml']
        with patch('sys.argv', testargs):
            cfgs = load_cfgs()
        self.assertTrue(len(cfgs) == 1)
        testargs = ['main', 'tests/dummy_configs/']
        with patch('sys.argv', testargs):
            cfgs = load_cfgs()
        self.assertTrue(len(cfgs) == 2)


if __name__ == "__main__":
    unittest.main()
