import unittest
import numpy as np


from src.datasets.dataset import dataset
from src.datasets.har import har


class test_dataset(unittest.TestCase):
    def test_create_float(self):
        dataset_card = har()
        train_data = dataset_card.train_data()
        test_data = dataset_card.test_data()
        train_labels = dataset_card.train_labels
        test_labels = dataset_card.test_labels
        # check if has exactly one unique value
        self.assertEqual(np.unique(train_data.dtypes.values).flatten().shape[0],1)
        # check if this unique value is np.float32
        self.assertEqual(np.unique(train_data.dtypes.values)[0], np.float32)
        # repeat for other objects
        self.assertEqual(np.unique(test_data.dtypes.values).flatten().shape[0],1)
        self.assertEqual(np.unique(test_data.dtypes.values)[0], np.float32)
        self.assertEqual(train_labels.dtypes, np.float32)
        self.assertEqual(test_labels.dtypes, np.float32)



    def test_load_datashapes(self):
        dataset_inst = dataset(
                name = 'dummy',
                file_path = './tests/dummy_dataset',
                )
        dataset_inst.load_used_ds()

        train_data = dataset_inst.train_data()
        train_label = dataset_inst.train_labels
        test_data = dataset_inst.test_data()
        test_label = dataset_inst.test_labels

        self.assertEqual(train_label.shape[0], train_data.shape[0])
        self.assertEqual(test_label.shape[0], test_data.shape[0])
        self.assertEqual(train_data.shape[1], test_data.shape[1])

    def test_save_datashapes(self):
        dataset_card = har()
        train_data = dataset_card.train_data()
        train_label = dataset_card.train_labels
        test_data = dataset_card.test_data()
        test_label = dataset_card.test_labels
        dataset_card.save('./tests/dummy_dataset_saved_and_loaded')

        dataset_inst = dataset(
                name = 'dummy2',
                file_path = './tests/dummy_dataset_saved_and_loaded',
                )
        train_data_2 = dataset_inst.train_data()
        train_label_2 = dataset_inst.train_labels
        test_data_2 = dataset_inst.test_data()
        test_label_2 = dataset_inst.test_labels
        self.assertTrue(not (False in np.unique((train_data_2 ==
            train_data).values)))
        self.assertTrue(not (False in np.unique((test_data_2 ==
            test_data).values)))
        self.assertTrue(not (False in np.unique((test_label_2 ==
            test_label).values)))
        self.assertTrue(not (False in np.unique((train_label_2 ==
            train_label).values)))

    def test_preprocess(self):
        dataset_card = har(subsample = 100)
        dataset_card.scale_min = -1
        dataset_card.scale_max = 1
        train_data = dataset_card.train_data()
        train_labels = dataset_card.train_labels
        self.assertTrue(np.unique(
            train_data.index == train_labels.index)[0])
        self.assertTrue(abs(train_data.min().min() + 1) < 0.00001)
        self.assertTrue(abs(train_data.max().max() - 1) < 0.00001)

    def test_calc_dists_to_label_mean(self):
        dataset_card = har()
        dists_to_label_mean = dataset_card.calc_dists_to_label_mean(subset =
                'train')



if __name__ == "__main__":
    unittest.main()
