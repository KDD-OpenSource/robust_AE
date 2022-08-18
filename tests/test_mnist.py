import unittest
import numpy as np


from src.datasets.mnist import mnist


class test_mnist(unittest.TestCase):
    def test_data_shapes(self):
        dataset = mnist()
        dataset.create()

        train_data = dataset.train_data()
        train_label = dataset.train_labels
        test_data = dataset.test_data()
        test_label = dataset.test_labels

        self.assertEqual(train_label.shape[0], train_data.shape[0])
        self.assertEqual(test_label.shape[0], test_data.shape[0])
        self.assertEqual(train_data.shape[1], test_data.shape[1])

    def test_normal_class(self):
        dataset = mnist(normal_class = 3)
        dataset.create()

        train_data = dataset.train_data()
        train_labels = dataset.train_labels
        test_data = dataset.test_data()
        test_labels = dataset.test_labels

        unique_test_labels = np.unique(test_labels.values)
        self.assertTrue(-1 in unique_test_labels)
        self.assertTrue(dataset.normal_class in unique_test_labels)
        self.assertEqual(len(unique_test_labels), 2)

        # assuming we train on normal data only 
        unique_train_labels = np.unique(train_labels.values)
        self.assertTrue(dataset.normal_class in unique_train_labels)
        self.assertEqual(len(unique_train_labels), 1)



if __name__ == "__main__":
    unittest.main()
