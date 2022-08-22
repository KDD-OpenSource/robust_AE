import unittest


from src.datasets.ionosphere import ionosphere


class test_ionosphere(unittest.TestCase):
    def test_data_shapes(self):
        dataset = ionosphere()
        dataset.create()

        train_data = dataset.train_data()
        train_label = dataset.train_labels
        test_data = dataset.test_data()
        test_label = dataset.test_labels

        self.assertEqual(train_label.shape[0], train_data.shape[0])
        self.assertEqual(test_label.shape[0], test_data.shape[0])
        self.assertEqual(train_data.shape[1], test_data.shape[1])


if __name__ == "__main__":
    unittest.main()
