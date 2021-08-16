import os
import time
import datetime
import copy
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from .neural_net import neural_net
from .neural_net import smallest_k_dist_loss
from .neural_net import IntermediateSequential


class autoencoder(neural_net):
    def __init__(
        self,
        topology: list,
        fct_dist: list,
        fct_dist_layer: int = None,
        denoising: bool = False,
        border_dist: bool = False,
        cross_point_sampling: bool = False,
        lambda_border: float = 0.01,
        lambda_fct: float = 0.1,
        name: str = "autoencoder",
        bias: bool = True,
        dropout: bool = False,
        L2Reg: float = 0,
        num_border_points: int = 1,
        num_epochs: int = 100,
        dynamic_epochs: bool = False,
        batch_size: int = 20,
        lr: float = 1e-3,
        seed: int = None,
        collect_subfcts=False,
    ):
        super().__init__(name, num_epochs, dynamic_epochs, batch_size, lr, seed)
        self.border_dist = border_dist
        self.cross_point_sampling = cross_point_sampling
        self.fct_dist = fct_dist
        if fct_dist_layer:
            self.fct_dist_layer = fct_dist_layer + 1
        else:
            self.fct_dist_layer = None
        self.denoising = denoising
        self.topology = topology
        self.lambda_border = lambda_border
        self.lambda_fct = lambda_fct
        self.bias = bias
        self.dropout = dropout
        self.L2Reg = L2Reg
        self.num_border_points = num_border_points
        self.collect_subfcts = collect_subfcts
        self.lin_sub_fct_Counters = []
        self.module = autoencoderModule(self.topology, self.bias, self.dropout)

    def fit(self, X: pd.DataFrame, run_folder=None, logger=None):
        data_loader = self.get_data_loader(X)
        # print(self.num_epochs)
        optimizer = torch.optim.Adam(
            params=self.module.parameters(), lr=self.lr, weight_decay=self.L2Reg
        )
        if self.collect_subfcts:
            self.lin_sub_fct_Counters.append(self.count_lin_subfcts(self.module, X))
        for epoch in range(self.num_epochs):
            if logger:
                logger.info(f"Training epoch {epoch}")
            epoch_start = datetime.datetime.now()
            self.module.train()
            epoch_loss = 0
            epoch_loss_reconstr = 0
            epoch_loss_border = 0
            epoch_loss_fct = 0
            i = 0
            data_len = len(data_loader)
            for inst_batch in data_loader:
                if self.denoising:
                    reconstr = self.module(
                        inst_batch
                        + np.random.normal(0, 0.01, size=inst_batch.shape).astype(
                            np.float32
                        )
                    )[0]
                else:
                    reconstr = self.module(inst_batch)[0]

                i = i + 1
                # print(i/data_len)
                loss_reconstr = nn.MSELoss()(inst_batch, reconstr)
                loss_border, loss_fct = smallest_k_dist_loss(
                    self.num_border_points,
                    border_dist=self.border_dist,
                    penal_dist=0.01,
                    fct_dist=self.fct_dist,
                    cross_point_sampling=self.cross_point_sampling,
                    max_layer=self.fct_dist_layer,
                )(inst_batch, self.module)
                loss = (
                    loss_reconstr
                    + self.lambda_border * loss_border
                    + self.lambda_fct * loss_fct
                )

                self.module.zero_grad()
                epoch_loss += loss
                epoch_loss_reconstr += loss_reconstr
                epoch_loss_border += loss_border
                epoch_loss_fct += loss_fct
                loss.backward()
                for layer_ind in range(len(self.module.get_neural_net())):
                    if isinstance(self.module.get_neural_net()[layer_ind], nn.Linear):
                        if (
                            self.module.get_neural_net()[layer_ind]
                            .weight.grad.isnan()
                            .sum()
                            > 0
                        ):
                            import pdb

                            pdb.set_trace()
                            loss_border, loss_fct = smallest_k_dist_loss(
                                self.num_border_points,
                                border_dist=self.border_dist,
                                fct_dist=self.fct_dist,
                            )(inst_batch, self.module)

                optimizer.step()
                # gc.collect()

            if run_folder:
                self.save(run_folder)
            if logger:
                logger.info(f"Epoch_loss: {epoch_loss}")
                logger.info(f"Epoch_loss_reconstr: {epoch_loss_reconstr}")
                logger.info(f"Epoch_loss_border: {epoch_loss_border}")
                logger.info(f"Epoch_loss_fct: {epoch_loss_fct}")
            self.module.eval()
            if self.collect_subfcts:
                self.lin_sub_fct_Counters.append(self.count_lin_subfcts(self.module, X))
            epoch_end = datetime.datetime.now()
            duration = epoch_end - epoch_start
            if logger:
                logger.info(f"Duration: {duration}")
        self.module.erase_dropout()
        if self.collect_subfcts:
            self.lin_sub_fct_Counters.append(self.count_lin_subfcts(self.module, X))

    def predict(self, X: pd.DataFrame):
        self.module.eval()
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        reconstructions = []
        for inst_batch in data_loader:
            inst_batch = inst_batch.float()
            reconstructions.append(self.module(inst_batch)[0].detach().numpy())
        reconstructions = np.vstack(reconstructions)
        reconstructions = pd.DataFrame(reconstructions, index=X.index)
        return reconstructions

    def param_cross_val(
        self, dataset, parameter, param_range, num_times=10, test_split=0.3
    ):
        X = dataset.train_data()
        param_lower_bound = param_range[0]
        param_upper_bound = param_range[1]
        current_best_param = None
        current_best_mean = None
        current_best_gaussian = None
        current_best_gaussian_var = None
        for param_value in np.linspace(param_lower_bound, param_upper_bound, num_times):
            # in order to parallelize one could:
            # create a module here (with the respective params)
            # do the train ... etc
            split = ShuffleSplit(n_splits=num_times, train_size=1 - test_split)
            self.__dict__[parameter] = param_value
            mean_error_list = []
            gaussian_nb_list = []
            for train_ds, test_ds in split.split(X):
                self.fit(X.loc[train_ds])
                res = self.predict(X.loc[test_ds])
                mean_error = ((X.loc[test_ds] - res) ** 2).sum(axis=1).mean()
                gnb_acc_latent = self.calc_naive_bayes(
                    X.loc[train_ds],
                    X.loc[test_ds],
                    dataset.train_labels[train_ds],
                    dataset.train_labels[test_ds],
                )
                gaussian_nb_list.append(gnb_acc_latent)
                mean_error_list.append(mean_error)
                self.module.apply(reset_weights)
            param_error_mean = np.array(mean_error_list).mean()
            gaussian_acc_mean = np.array(gaussian_nb_list).mean()
            gaussian_acc_var = np.var(np.array(gaussian_nb_list))
            print(f"Param_value: {param_value}")
            print(f"Gaussian: {gaussian_acc_mean}")
            print(f"GaussianVar: {gaussian_acc_var}")
            if current_best_mean is None or gaussian_acc_mean > current_best_gaussian:
                current_best_mean = param_error_mean
                current_best_param = param_value
                current_best_gaussian = gaussian_acc_mean
                current_best_gaussian_var = gaussian_acc_var
        return (
            current_best_param,
            current_best_mean,
            current_best_gaussian,
            current_best_gaussian_var,
        )

    def calc_naive_bayes(self, train_data, test_data, train_labels, test_labels):
        latent_repr_train = self.extract_latent(train_data)
        latent_repr_test = self.extract_latent(test_data)

        gaussian_latent = GaussianNB()
        gaussian_latent.fit(latent_repr_train, train_labels)
        latent_res = gaussian_latent.predict(latent_repr_test)
        acc_latent = accuracy_score(test_labels, latent_res)
        return acc_latent

    def extract_latent(self, X: pd.DataFrame):
        self.module.eval()
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        latent_repr = []
        latent_layer_key = str(int(len(self.module.get_neural_net()) / 2))
        for inst_batch in data_loader:
            inst_batch = inst_batch.float()
            latent_repr.append(
                self.module(inst_batch)[1][latent_layer_key].detach().numpy()
            )
        latent_repr = np.vstack(latent_repr)
        latent_repr = pd.DataFrame(latent_repr)
        return latent_repr

    def save(self, path):
        # path is the folder within reports in which it has been trained
        os.makedirs(os.path.join("./models/trained_models", self.name), exist_ok=True)
        torch.save(
            {
                "topology": self.topology,
                "denoising": self.denoising,
                "cross_point_sampling": self.cross_point_sampling,
                "fct_dist": self.fct_dist,
                "fct_dist_layer": self.fct_dist_layer,
                "border_dist": self.border_dist,
                "lambda_border": self.lambda_border,
                "lambda_fct": self.lambda_fct,
                "num_border_points": self.num_border_points,
                "bias": self.bias,
                "dropout": self.dropout,
                "L2Reg": self.L2Reg,
                "collect_subfcts": self.collect_subfcts,
                "lin_sub_fct_Counters": self.lin_sub_fct_Counters,
            },
            os.path.join(path, "model_detailed.pth"),
        )

        torch.save(
            {
                "topology": self.topology,
                "denoising": self.denoising,
                "cross_point_sampling": self.cross_point_sampling,
                "fct_dist": self.fct_dist,
                "fct_dist_layer": self.fct_dist_layer,
                "border_dist": self.border_dist,
                "lambda_border": self.lambda_border,
                "lambda_fct": self.lambda_fct,
                "num_border_points": self.num_border_points,
                "bias": self.bias,
                "dropout": self.dropout,
                "L2Reg": self.L2Reg,
                "collect_subfcts": self.collect_subfcts,
                "lin_sub_fct_Counters": self.lin_sub_fct_Counters,
            },
            os.path.join("./models/trained_models", self.name, "model_detailed.pth"),
        )

        torch.save(self.module.state_dict(), os.path.join(path, "model.pth"))
        torch.save(
            self.module.state_dict(),
            os.path.join("./models/trained_models", self.name, "model.pth"),
        )

    def load(self, path):
        model_details = torch.load(os.path.join(path, "model_detailed.pth"))

        self.topology = model_details["topology"]
        self.denoising = model_details["denoising"]
        self.cross_point_sampling = model_details["cross_point_sampling"]
        # this code should render the if statements below useless
        #import pdb; pdb.set_trace()
        self.fct_dist = model_details["fct_dist"]
        self.fct_dist_layer = model_details["fct_dist_layer"]
        self.border_dist = model_details["border_dist"]
        self.num_border_points = model_details["num_border_points"]
        self.bias = model_details["bias"]
        self.dropout = model_details["dropout"]
        self.L2Reg = model_details["L2Reg"]
        self.collect_subfcts = model_details["collect_subfcts"]
        self.lin_sub_fct_Counters = model_details["lin_sub_fct_Counters"]
        self.lambda_border = model_details["lambda_border"]
        self.lambda_fct = model_details["lambda_fct"]
        if self.fct_dist != model_details["fct_dist"]:
            print(
                f"""
                You test with fct_dist being {self.fct_dist} but you trained
                with fct_dist being {model_details["fct_dist"]}."""
            )
        if self.border_dist != model_details["border_dist"]:
            print(
                f"""
                You test with border_dist being {self.border_dist} but you trained
                with border_dist being {model_details["border_dist"]}."""
            )
        if self.num_border_points != model_details["num_border_points"]:
            print(
                f"""
                You test with num_border_points being {self.num_border_points} but you trained
                with num_border_points being {model_details["num_border_points"]}."""
            )

        # note that we hardcode dropout to False as we do not want to have
        # dropout in testing (just in Training)
        self.module = autoencoderModule(self.topology, self.bias, dropout=False)
        self.module.load_state_dict(torch.load(os.path.join(path, "model.pth")))

    def assign_errors(self, neural_net_mod, X):
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        sample_error_pairs = []
        ctr = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                reconstr = neural_net_mod(inst)[0]
                error = nn.MSELoss()(inst, reconstr).detach().numpy()
                sample_error_pairs.append((inst, error))
        return sample_error_pairs

    def calc_errors(self, neural_net_mod, X):
        errors = pd.Series(0, X.index)
        for ind in X.index:
            inst = torch.tensor(X.loc[ind])
            reconstr = neural_net_mod(inst)[0]
            error = nn.MSELoss()(inst, reconstr).detach().numpy()
            errors.loc[ind] = error
        return errors


class autoencoderModule(nn.Module):
    def __init__(self, topology: list, bias: bool = True, dropout: bool = False):
        super().__init__()
        layers = np.array(topology).repeat(2)[1:-1]
        if len(topology) % 2 == 0:
            print(layers)
            raise Warning(
                """Your topology is probably not what you want as the
                    hidden layer is repeated multiple times"""
            )
        if dropout:
            nn_layers = np.array(
                [
                    [nn.Linear(int(a), int(b)), nn.ReLU(), nn.Dropout(p=0.1)]
                    for a, b in layers.reshape(-1, 2)
                ]
            ).flatten()[:-2]
        else:
            nn_layers = np.array(
                [
                    [nn.Linear(int(a), int(b)), nn.ReLU()]
                    for a, b in layers.reshape(-1, 2)
                ]
            ).flatten()[:-1]
        self._neural_net = IntermediateSequential(*nn_layers)

    def forward(self, inst_batch, return_act: bool = True):
        reconstruction, intermediate_outputs = self._neural_net(inst_batch)
        return reconstruction, intermediate_outputs

    def get_neural_net(self):
        return self._neural_net

    def erase_dropout(self):
        new_layers = []
        for layer in self.get_neural_net():
            if not isinstance(layer, nn.Dropout):
                new_layers.append(layer)
        self._neural_net = IntermediateSequential(*new_layers)


def reset_weights(layer):
    if isinstance(layer, nn.Linear):
        layer.reset_parameters()
