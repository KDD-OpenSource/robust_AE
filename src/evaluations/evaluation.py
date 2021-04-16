import os
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd


class evaluation:
    def __init__(self):
        self.result_folder = os.path.join(os.getcwd(), "reports")

    def make_run_folder(self, ctx):
        datetime = time.strftime("%Y-%m-%dT%H:%M:%S")
        folder_name = datetime + "_" + ctx
        self.run_folder = os.path.join(self.result_folder, folder_name)
        os.mkdir(self.run_folder)

    def get_run_folder(self):
        return self.run_folder

    def save_figure(self, figure, name: str, subfolder=None):
        name = name + ".png"
        if subfolder:
            os.makedirs(os.path.join(self.run_folder, subfolder), exist_ok=True)
            figure.savefig(os.path.join(self.run_folder, subfolder, name))
        else:
            figure.savefig(os.path.join(self.run_folder, name))

    def save_csv(self, data, name: str):
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        name = name + ".csv"
        data.to_csv(os.path.join(self.run_folder, name))
