import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd

from .neural_net import neural_net
from .neural_net import IntermediateSequential


class autoencoder(neural_net):
    def __init__(
        self,
        topology: list,
        name: str = "autoencoder",
        bias: bool = True,
        dropout: bool = False,
        num_epochs: int = 100,
        batch_size: int = 10,
        lr: float = 1e-3,
        seed: int = None,
        collect_subfcts=False,
    ):
        super().__init__(name, num_epochs, batch_size, lr, seed)
        self.topology = topology
        self.bias = bias
        self.dropout = dropout
        self.collect_subfcts = collect_subfcts
        self.lin_sub_fct_Counters = []
        self.module = autoencoderModule(self.topology, self.bias, self.dropout)

    def fit(self, X: pd.DataFrame):
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        optimizer = torch.optim.Adam(params=self.module.parameters(), lr=self.lr)
        if self.collect_subfcts:
            self.lin_sub_fct_Counters.append(self.count_lin_subfcts(self.module, X))
        for epoch in range(self.num_epochs):
            self.module.train()
            epoch_loss = 0
            for inst_batch in data_loader:

                # this code is just for code testing
                # for inst in inst_batch:
                # self.lrp_ae(self.module, inst)
                # self.get_fctBoundary(self.module, inst)
                # self.get_closest_funcBoundary(self.module, inst)

                inst_batch = inst_batch.float()
                reconstr = self.module(inst_batch)[0]
                loss = nn.MSELoss()(inst_batch, reconstr)
                self.module.zero_grad()
                epoch_loss += loss
                loss.backward()
                optimizer.step()

            print(epoch_loss)
            self.module.eval()
            if self.collect_subfcts:
                self.lin_sub_fct_Counters.append(self.count_lin_subfcts(self.module, X))
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
        reconstructions = pd.DataFrame(reconstructions)
        return reconstructions

    def save(self, path):
        # path is the folder within reports in which it has been trained
        os.makedirs(os.path.join('./models', self.name), exist_ok=True)
        torch.save(
            {
                "topology": self.topology,
                "bias": self.bias,
                "dropout": self.dropout,
                "collect_subfcts": self.collect_subfcts,
                "lin_sub_fct_Counters": self.lin_sub_fct_Counters
            },
            os.path.join(path, "model_detailed.pth"),
        )

        torch.save(
            {
                "topology": self.topology,
                "bias": self.bias,
                "dropout": self.dropout,
                "collect_subfcts": self.collect_subfcts,
                "lin_sub_fct_Counters": self.lin_sub_fct_Counters
            },
            os.path.join("./models", self.name, "model_detailed.pth"),
        )

        torch.save(self.module.state_dict(), os.path.join(path, "model.pth"))
        torch.save(self.module.state_dict(), os.path.join("./models",
            self.name, "model.pth"))

    def load(self, path):
        model_details = torch.load(os.path.join(path, "model_detailed.pth"))

        self.topology = model_details["topology"]
        self.bias = model_details["bias"]
        self.dropout = model_details["dropout"]
        self.collect_subfcts = model_details["collect_subfcts"]
        self.lin_sub_fct_Counters = model_details["lin_sub_fct_Counters"]

        self.module = autoencoderModule(self.topology, self.bias, self.dropout)
        self.module.load_state_dict(torch.load(os.path.join(path, "model.pth")))




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
        reconstruction, intermediate_outputs = self._neural_net(inst_batch.float())
        return reconstruction, intermediate_outputs

    def get_neural_net(self):
        return self._neural_net

    def erase_dropout(self):
        new_layers = []
        for layer in self.get_neural_net():
            if not isinstance(layer, nn.Dropout):
                new_layers.append(layer)
        self._neural_net = IntermediateSequential(*new_layers)
