from pprint import pprint

import abc
import copy
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from functools import reduce


class neural_net:
    def __init__(
        self,
        name: str,
        num_epochs: int = 100,
        batch_size: int = 20,
        lr: float = 1e-3,
        seed: int = None,
    ):
        self.name = name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = float(lr)
        self.seed = seed

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    def get_lin_subfct(self, neural_net_mod, instance, max_layer = None):
        if max_layer is not None:
            neural_net = copy.deepcopy(
                    neural_net_mod.get_neural_net()[:max_layer])
        else:
            neural_net = copy.deepcopy(neural_net_mod.get_neural_net())
        #import pdb; pdb.set_trace()
        _, intermed_results = neural_net_mod(instance)
        linear_layers_list = []
        bias_list = []
        for key, value in intermed_results.items():
            if (
                int(key) < len(neural_net) - 1
                and isinstance(neural_net[int(key)], nn.Linear)
                and isinstance(neural_net[int(key) + 1], nn.ReLU)
            ):
                reluKey = str(int(key) + 1)
                matrix = neural_net[int(key)].weight.detach().numpy()
                bias = neural_net[int(key)].bias.detach().numpy()
                for value_ind in range(len(intermed_results[reluKey])):
                    if intermed_results[reluKey][value_ind] == 0:
                        matrix[value_ind] = 0
                        bias[value_ind] = 0
                linear_layers_list.append(matrix)
                bias_list.append(bias)
            # go through all layers
        if isinstance(neural_net[-1], nn.Linear):
            linear_layers_list.append(neural_net[-1].weight.detach().numpy())
            bias_list.append(neural_net[-1].bias.detach().numpy())
        linear_layers_transposed = list(
            map(lambda x: x.transpose(), linear_layers_list)
        )
        result_matrix = reduce(np.matmul, linear_layers_transposed)

        result_bias = bias_list[-1]
        for bias_ind in range(len(bias_list) - 2, -1, -1):
            bias_matrix = reduce(np.matmul, linear_layers_transposed[bias_ind + 1 :])
            result_bias += np.matmul(bias_list[bias_ind], bias_matrix)
        result_matrix = result_matrix.transpose()
        lin_subfct = linearSubfunction(result_matrix, result_bias)
        return lin_subfct

    def check_final_linFct(self, neural_net_mod, instance):
        _, intermed_results = neural_net_mod(instance)
        lin_subfct = self.get_lin_subfct(neural_net_mod, instance)
        mat = lin_subfct.matrix
        bias = lin_subfct.bias
        linFctRes = np.matmul(mat, instance) + bias
        linFctRes = linFctRes.float()
        output_key = list(intermed_results.keys())[-1]
        acc = 1e-6
        if (
            (abs((linFctRes - intermed_results[output_key]) / linFctRes)) > acc
        ).sum() == 0:
            return 0
        else:
            return 1

    def check_interm_linFcts(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        _, intermed_results = neural_net_mod(instance)

        linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        linLayers = self.get_linLayers(neural_net_mod)
        mats = list(map(lambda x: x.matrix, linSubFcts))
        biases = list(map(lambda x: x.bias, linSubFcts))
        num_inacc = 0
        for fct_ind in range(len(mats)):
            linFctRes = np.matmul(mats[fct_ind], instance) + biases[fct_ind]
            linFctRes = linFctRes.float()
            output_key = str(linLayers[fct_ind])
            acc = 1e-6
            if (
                (abs((linFctRes - intermed_results[output_key]) / linFctRes)) > acc
            ).sum() != 0:
                num_inacc += 1

        return num_inacc

    def count_lin_subfcts(self, neural_net_mod, X):
        functions = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        for inst_batch in data_loader:
            for inst in inst_batch:
                functions.append(self.get_lin_subfct(neural_net_mod, inst))
        unique_functions = []
        functions_counter = []
        # there is probably a smarter way to do this?
        for function in functions:
            if function not in unique_functions:
                unique_functions.append(function)
                functions_counter.append(1)
            else:
                index = unique_functions.index(function)
                functions_counter[index] += 1
        return list(zip(unique_functions, functions_counter))

    def get_points_of_linsubfcts(self, neural_net_mod, X):
        functions = {}
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        for inst_batch in data_loader:
            for inst in inst_batch:
                function = self.get_lin_subfct(neural_net_mod, inst)
                if function not in functions.keys():
                    functions[function] = []
                else:
                    functions[function].append(inst)
        return functions

    def get_signature(self, neural_net_mod, instance):
        intermed_res = neural_net_mod(instance)[1]
        layer_key_list = list(zip(list(neural_net_mod.get_neural_net()),
            intermed_res))
        relu_key_list = []
        for layer in layer_key_list:
            if isinstance(layer[0], nn.ReLU):
                relu_key_list.append(layer[1])
        signature_dict = {}
        for key in relu_key_list:
            signature_dict[key] = intermed_res[key]
        for key, value in signature_dict.items():
            signature_dict[key] = self.binarize(value)
        return signature_dict

    def binarize(self, tensor):
        nparray = tensor.detach().numpy()
        npmask = nparray > 0
        nparray[npmask] = 1
        return nparray

    def get_pot_neuronBoundary(self, neural_net_mod, instance, neuron):
        pass

    def get_interm_linFct(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        linSubFcts = []
        linLayers = self.get_linLayers(neural_net_mod)
        for layer_ind in linLayers:
            linSubFcts.append(self.get_lin_subfct(neural_net_mod,
                instance, layer_ind+1))
        return linSubFcts

    def get_linLayers(self, neural_net_mod):
        linLayers = []
        for layer_ind in range(len(list(neural_net_mod.get_neural_net()))):
            if isinstance(neural_net_mod.get_neural_net()[layer_ind],
                    nn.Linear):
                linLayers.append(layer_ind)
        return linLayers

    def get_closest_funcBoundary(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        for layer in linSubFcts:
            for neuron_ind in range(len(layer.matrix)):
                normal_vector = (layer.matrix[neuron_ind]
                        /np.linalg.norm(layer.matrix[neuron_ind]))
                dist_to_border = (np.matmul(normal_vector, instance)
                    + layer.bias[neuron_ind])
                cross_points.append((instance - dist_to_border *
                    normal_vector))
                if cross_points[-1].isnan().sum()>0:
                    print(cross_points[-1])
                    import pdb; pdb.set_trace()
                #check = (np.matmul(normal_vector, border_point) +
                    #layer.bias[neuron_ind])
        point_signatures = []
        sig_counter = 0
        for point in cross_points:
            point_signatures.append(self.get_signature(neural_net_mod,
                point))
            if self.isequal_sig_dict(instance_sig, point_signatures[-1]):
                sig_counter += 1

    def isequal_sig_dict(self, signature1, signature2):
        if signature1.keys() != signature2.keys():
            return False
        keys = signature1.keys()
        return all(np.array_equal(signature1[key], signature2[key]) for key in
                keys)

    def erase_ReLU(self, neural_net_mod):
        new_layers = []
        for layer in neural_net_mod.get_neural_net():
            if not isinstance(layer, nn.ReLU):
                new_layers.append(layer)
        return IntermediateSequential(*new_layers)

    def interpolate(point_from: torch.tensor, point_to: torch.tensor, num_steps):
        inter_points = [point_from + int_var/(num_steps-1) * point_to for int_var in
                range(num_steps)]
        return torch.stack(inter_points)

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class linearSubfunction:
    def __init__(self, matrix, bias):
        self.matrix = matrix
        self.bias = bias
        self.name = self.__hash__()

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        return (self.matrix == other.matrix).all() and (self.bias == other.bias).all()

    def __key(self):
        return (self.matrix.tobytes(), self.bias.tobytes())

    def __hash__(self):
        return hash(self.__key())
