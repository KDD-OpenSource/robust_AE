import unittest
from tests.test_dataset import test_dataset
from tests.test_moteStrain import test_moteStrain
from tests.test_sonyAIBORobotSurface1 import test_sonyAIBORobotSurface1
from tests.test_sonyAIBORobotSurface2 import test_sonyAIBORobotSurface2
from tests.test_twoLeadEcg import test_twoLeadEcg
from tests.test_phalangesOutlinesCorrect import test_phalangesOutlinesCorrect
from tests.test_proximalPhalanxOutlineCorrect import test_proximalPhalanxOutlineCorrect
from tests.test_sineNoise import test_sineNoise
from tests.test_mnist import test_mnist
from tests.test_page_blocks import test_page_blocks
from tests.test_segment import test_segment
from tests.test_steel_plates_fault import test_steel_plates_fault
from tests.test_wbc import test_wbc
from tests.test_satellite import test_satellite
from tests.test_qsar_biodeg import test_qsar_biodeg
from tests.test_gas_drift import test_gas_drift
from tests.test_har import test_har



def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(test_dataset))
    suite.addTest(unittest.makeSuite(test_moteStrain))
    suite.addTest(unittest.makeSuite(test_sonyAIBORobotSurface1))
    suite.addTest(unittest.makeSuite(test_sonyAIBORobotSurface2))
    suite.addTest(unittest.makeSuite(test_twoLeadEcg))
    suite.addTest(unittest.makeSuite(test_phalangesOutlinesCorrect))
    suite.addTest(unittest.makeSuite(test_proximalPhalanxOutlineCorrect))
    suite.addTest(unittest.makeSuite(test_sineNoise))
    suite.addTest(unittest.makeSuite(test_mnist))
    suite.addTest(unittest.makeSuite(test_page_blocks))
    suite.addTest(unittest.makeSuite(test_segment))
    suite.addTest(unittest.makeSuite(test_steel_plates_fault))
    suite.addTest(unittest.makeSuite(test_wbc))
    suite.addTest(unittest.makeSuite(test_satellite))
    suite.addTest(unittest.makeSuite(test_qsar_biodeg))
    suite.addTest(unittest.makeSuite(test_gas_drift))
    suite.addTest(unittest.makeSuite(test_har))
    return suite



if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity = 2)
    test_suite = suite()
    runner.run(test_suite)
