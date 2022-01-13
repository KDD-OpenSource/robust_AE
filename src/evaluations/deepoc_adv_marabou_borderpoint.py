import maraboupy
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import time
from maraboupy import Marabou
from maraboupy import MarabouCore
from itertools import combinations

from .evaluation import evaluation


class deepoc_adv_marabou_borderpoint:
    def __init__(
        self,
        eval_inst: evaluation,
        name: str = "deepoc_adv_marabou_borderpoint",
        # accuracy wrt distance in input space
        accuracy=0.0001,
    ):
        self.name = name
        self.evaluation = eval_inst
        self.accuracy = accuracy

    # general plan: 
    # take random input point (-> mean? a set of? etc.)
    # calculate image
    # calculate border point (we need R for this, where to get this?)
    # find preimage

    def calc_border_point(self, point, algorithm):
        pass

    def get_samples(self, sampling_method: str, points=None):
        pass

    def find_closest_preimage(self, border_point, algorithm):
        pass


    def evaluate(self, dataset, algorithm):
        import pdb; pdb.set_trace()
        collapsing = test_for_collapsing(dataset, algorithm)
        network = self.get_network(algorithm, dataset)
        samples = self.get_samples('random_points', points = 3)
        result_dict = {}
        for i,sample in enumerate(samples):
            output_sample = algorithm.predict(input_sample)
            sample_border_point = self.calc_border_point(output_sample,
                    algorithm)
            closest_preimage = self.find_closest_preimage(sample_border_point,
                    algorithm)
            result_dict[i] = (sample, closest_preimage)
        self.evaluation.save_json(result_dict, "deepoc_marabou_closest_adv")

    def get_network(self, algorithm, dataset):
        randomInput = torch.randn(1, algorithm.topology[0])
        run_folder = self.evaluation.run_folder[
            self.evaluation.run_folder.rfind("202") :
        ]
        onnx_folder = os.path.join(
            "./models/onnx_models/",
            str(algorithm.name) + "_" + dataset.name,
            run_folder,
        )
        marabou_folder = os.path.join(
            "./models/marabou_models/",
            str(algorithm.name) + "_" + dataset.name,
            run_folder,
        )
        os.makedirs(onnx_folder, exist_ok=True)
        os.makedirs(marabou_folder, exist_ok=True)
        torch.onnx.export(
            algorithm.module.get_neural_net(),
            randomInput.float(),
            os.path.join(onnx_folder, "saved_algorithm.onnx"),
        )
        network = Marabou.read_onnx(
            os.path.join(onnx_folder, "saved_algorithm.onnx"),
            outputName=str(2 * len(algorithm.module.get_neural_net()) + 1),
        )
        return network


def test_for_collapsing(dataset, algorithm):
    pred_dataset = algorithm.predict(dataset.test_data())
    if pred_dataset.var().sum() < 0.00001:
        return True
    else:
        return False
