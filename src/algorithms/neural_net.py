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

    def get_lin_subfct(self, neural_net_mod, instance, max_layer=None):
        #import pdb; pdb.set_trace()
        if max_layer is not None:
            neural_net = copy.deepcopy(neural_net_mod.get_neural_net()[:max_layer])
        else:
            neural_net = copy.deepcopy(neural_net_mod.get_neural_net())
        # import pdb; pdb.set_trace()
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

    def assign_lin_subfcts(self, neural_net_mod, X):
        functions = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        inst_func_pairs = []
        for inst_batch in data_loader:
            for inst in inst_batch:
                inst_func_pairs.append(
                    (inst, self.get_lin_subfct(neural_net_mod, inst))
                )
        return inst_func_pairs

    def assign_border_dists(self, neural_net_mod, X):
        dists = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        sample_dist_pairs = []
        ctr = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                sample_dist_pairs.append(
                    (inst, self.get_closest_funcBoundary(neural_net_mod,
                        inst)[1])
                )
                ctr += 1
                #print(ctr)
        return sample_dist_pairs

    def assign_lin_subfcts_ind(self, neural_net_mod, X):
        functions = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=True,
            pin_memory=True,
        )
        inst_func_pairs = []
        for inst_batch in data_loader:
            for inst in inst_batch:
                inst_func_pairs.append(
                    (inst, self.get_lin_subfct(neural_net_mod, inst))
                )

        unique_functions = []
        functions_counter = []
        inst_ind_pairs = []
        # there is probably a smarter way to do this?
        for inst, function in inst_func_pairs:
            if function not in unique_functions:
                unique_functions.append(function)
                functions_counter.append(1)
                inst_ind_pairs.append((inst, len(functions_counter)))
            else:
                index = unique_functions.index(function)
                # functions_counter[index] += 1
                inst_ind_pairs.append((inst, index))
        return inst_ind_pairs

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
        layer_key_list = list(zip(list(neural_net_mod.get_neural_net()), intermed_res))
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
            linSubFcts.append(
                self.get_lin_subfct(neural_net_mod, instance, layer_ind + 1)
            )
        return linSubFcts

    def get_linLayers(self, neural_net_mod):
        linLayers = []
        for layer_ind in range(len(list(neural_net_mod.get_neural_net()))):
            if isinstance(neural_net_mod.get_neural_net()[layer_ind], nn.Linear):
                linLayers.append(layer_ind)
        return linLayers

    def get_closest_funcBoundary(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        ####linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_point = self.get_cross_point(neuron_subFct, instance)
            if cross_point is not None:
                cross_points.append(cross_point)
        if None in cross_points:
            print('NoneType in cross_points')
            return None
        if len(cross_points) == 0:
            print('No cross_points found')
            return None
        diffs = torch.stack(cross_points) - instance
        dists = np.linalg.norm(np.array(diffs), axis=1)
        cross_points_sorted = sorted(list(zip(dists, cross_points)))
        for dist_cross_point_pair in cross_points_sorted:
            if self.check_true_cross_point(neural_net_mod, instance,
                    dist_cross_point_pair[1]):
                return (dist_cross_point_pair[1],
            dist_cross_point_pair[0])

    def check_true_cross_point(self, neural_net_mod, instance, cross_point):
        inst_linFct = self.get_lin_subfct(neural_net_mod, instance)
        cross_point_linFct = self.get_lin_subfct(neural_net_mod, cross_point)
        before_cross = instance + 0.99 * (cross_point - instance)
        after_cross = instance + 1.01 * (cross_point - instance)
        before_cross_linFct = self.get_lin_subfct(neural_net_mod, before_cross)
        after_cross_linFct = self.get_lin_subfct(neural_net_mod, after_cross)
        if (inst_linFct == before_cross_linFct and inst_linFct !=
        after_cross_linFct):
            return True
        else:
            if inst_linFct != before_cross_linFct:
                print('CrossPoint has different function')
            if inst_linFct == after_cross_linFct:
                print('After crossing it has still the same function')
            print('False')
            return False

    def get_closest_funcBoundaries(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_points.append(self.get_cross_point(neuron_subFct, instance))
        #for layer in linSubFcts:
        #    for neuron_ind in range(len(layer.matrix)):
        #        normal_vector = layer.matrix[neuron_ind] / np.linalg.norm(
        #            layer.matrix[neuron_ind]
        #        )
        #        dist_to_border = (
        #            np.matmul(layer.matrix[neuron_ind], instance)
        #            + layer.bias[neuron_ind]
        #        ) / np.linalg.norm(layer.matrix[neuron_ind])
        #        # dist_to_border = (
        #        #   np.matmul(normal_vector, instance) + layer.bias[neuron_ind]
        #        # )
        #        cross_points.append((instance - dist_to_border * normal_vector))
        #        if cross_points[-1].isnan().sum() > 0:
        #            print(cross_points[-1])
        #            import pdb

        #            pdb.set_trace()
        #        # check = (np.matmul(normal_vector, border_point) +
        #        # layer.bias[neuron_ind])
        #point_signatures = []
        #sig_counter = 0

        # redo the next part
        closest_cross_points = []
        for point in cross_points:
            point_signatures.append(self.get_signature(neural_net_mod, point))
            if isequal_sig_dict(instance_sig, point_signatures[-1]):
                sig_counter += 1
                closest_cross_points.append(point)
        return closest_cross_points

    def get_all_funcBoundaries(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
        #linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        # calculate crosspoints
        cross_points = []
        neuron_subFcts = self.get_neuron_border_subFcts(neural_net_mod, instance)
        for neuron_subFct in neuron_subFcts:
            cross_points.append(self.get_cross_point(neuron_subFct, instance))
        return cross_points

    def erase_ReLU(self, neural_net_mod):
        new_layers = []
        for layer in neural_net_mod.get_neural_net():
            if not isinstance(layer, nn.ReLU):
                new_layers.append(layer)
        return IntermediateSequential(*new_layers)

    def interpolate(point_from: torch.tensor, point_to: torch.tensor, num_steps):
        inter_points = [
            point_from + int_var / (num_steps - 1) * point_to
            for int_var in range(num_steps)
        ]
        return torch.stack(inter_points)

    def get_neuron_subFcts(self, neural_net_mod, instance):
        linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        neuron_subFcts = []
        for layer in linSubFcts:
            neuron_subFcts.extend(self.get_neuron_subFcts_from_layer(layer))
        return neuron_subFcts

    def get_neuron_border_subFcts(self, neural_net_mod, instance):
        linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        neuron_subFcts = []
        for layer in linSubFcts[:-1]:
            neuron_subFcts.extend(self.get_neuron_subFcts_from_layer(layer))
        return neuron_subFcts

    def get_neuron_subFcts_from_layer(self, layer_subfunction):
        neuron_subFcts = []
        for neuron_ind in range(len(layer_subfunction.matrix)):
            neuron_subFcts.append((layer_subfunction.matrix[neuron_ind],
                layer_subfunction.bias[neuron_ind]))
        return neuron_subFcts

    def get_neuron_subFct_cross_point(self, neural_net_mod, instance):
        # maybe this function is irrelevant...
        neural_net_mod = copy.deepcopy(neural_net_mod)
        linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        subFct_cross_points = []
        for layer in linSubFcts:
            for neuron_ind in range(len(layer.matrix)):
                neuron_sub_fct = (layer.matrix[neuron_ind],
                        layer.bias[neuron_ind])
                cross_point = self.get_cross_point(neuron_sub_fct, instance)
                subFct_cross_points.append(
                        (neuron_sub_fct, cross_point)
                        )
        # result has the followin structure:
        # ((fct_vector, bias), cross_point)
        return subFct_cross_points


    def get_cross_point(self, neuron, instance):
        normal_vector = neuron[0] / np.linalg.norm(
            neuron[0]
        )
        dist_to_border = (
            np.matmul(neuron[0], instance)
            + neuron[1]
        ) / np.linalg.norm(neuron[0])
        cross_point = (instance - dist_to_border * normal_vector)
        if cross_point.isnan().sum() > 0:
            # is this possible?
            return None
        else:
            return cross_point
        # neuron is a tuple consisting of weights and the bias

    def check_interp_signatures(self, point_from, point_to, num_steps, neural_net_mod):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        inter_points = neural_net.interpolate(point_from, point_to, num_steps)
        point_from_sig = self.get_signature(neural_net_mod, point_from)
        for point in inter_points:
            point_inter_sig = self.get_signature(neural_net_mod, point)
            if isequal_sig_dict(point_from_sig, point_inter_sig) == False:
                return False
        return True

    def get_fctBoundary(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        closest_boundaries = self.get_closest_funcBoundaries(
            self, neural_net_mod, instance
        )
        # sample points defined by hyperplanes
        # find counterexample in points
        # interpolate between inst and counterexample to find another boundary
        # (binary search)
        # iterate until with 10k sample we find no inlier
        pass

    def get_fct_area(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        #boundaries = self.get_all_funcBoundaries(self, neural_net_mod, instance)
        #linSubFcts = self.get_interm_linFct(neural_net_mod, instance)
        #neuron_subFcts = self.get_neuron_subFcts(neural_net_mod, instance) 
        neuron_subFcts_cross_points = self.get_neuron_subFct_cross_point(
                neural_net_mod, instance)
        #for layer in linSubFcts:
            #neuron_subFcts.extend(self.get_neuron_subFcts(layer))
        limiting_neurons = []
        inst_lin_subFct = self.get_lin_subfct(neural_net_mod, instance)
        for neuron_sub_fct, cross_point in neuron_subFcts_cross_points:
            # check if the following equation will be right
            crossed_inst = instance + 1.1 * (cross_point - instance)
            crossed_inst_lin_subFct = self.get_lin_subfct(neural_net_mod,
                    crossed_inst)
            if inst_lin_subFct != crossed_inst_lin_subFct:
                limiting_neurons.append(neuron_sub_fct)
        import pdb; pdb.set_trace()
        return limiting_neurons



        # for each potential boundary
        # check if in set of real boundaries (consider cases)
            # cases:
            # 1) within same linear function -> check if fct changes on
            # sampling line
            # 2) directly reachable border -> check if fct changes on exactly the point
            # 3) indirectly reachable border ->
            # 4) irrelevant border (not restricting the current set)
        # add to real boundaries (in form of an equation) to be checked.

        # a simpler way is just to find out the points in the same function
        # that remain on the same function... For the rest, just collect all
        # the equations... (can we have a 'strip_redundant_border function?')


    def lrp_ae(self, neural_net_mod, instance, gamma=1):
        input_relevance = np.zeros(instance.size())
        neural_net_mod = copy.deepcopy(neural_net_mod)
        output, intermed_res = neural_net_mod(instance)
        layer_inds = range(len(neural_net_mod.get_neural_net()))
        gamma = gamma
        error = nn.MSELoss()(instance, output)
        relevance = torch.tensor((np.array(instance) - output.detach().numpy()) ** 2)

        relevance_bias = 0
        for layer_ind in layer_inds[:0:-1]:
            layer = neural_net_mod.get_neural_net()[layer_ind]
            activation = intermed_res[str(layer_ind - 1)]
            if isinstance(layer, nn.Linear):
                if layer_ind != (len(layer_inds) - 1) and isinstance(
                    neural_net_mod.get_neural_net()[layer_ind + 1], nn.ReLU
                ):
                    relevance = self.lrp_linear_relu(
                        activation, layer, relevance, gamma
                    )
                    relevance_bias += relevance[-1]
                    relevance = relevance[:-1]
                else:
                    relevance = self.lrp_linear(activation, layer, relevance)
                    relevance_bias += relevance[-1]
                    relevance = relevance[:-1]
        return relevance

    def lrp_linear(self, activation, layer, relevance):
        act_bias = torch.cat((activation, torch.tensor([1])))
        layer_weight = torch.cat((layer.weight, layer.bias.reshape(-1, 1)), dim=1)
        relevance = (
            (
                (act_bias * layer_weight).transpose(0, 1)
                / ((act_bias * layer_weight).sum(axis=1))
            )
            * relevance
        ).sum(axis=1)
        return relevance

    def lrp_linear_relu(self, activation, layer, relevance, gamma):
        pos_weights = copy.deepcopy(layer.weight)
        pos_weights.detach()[pos_weights < 0] = 0
        act_bias = torch.cat((activation, torch.tensor([1])))
        layer_weight = torch.cat((layer.weight, layer.bias.reshape(-1, 1)), dim=1)
        pos_layer_weight = torch.cat((pos_weights, layer.bias.reshape(-1, 1)), dim=1)
        relevance = (
            (
                (act_bias * (layer_weight + gamma * pos_layer_weight)).transpose(0, 1)
                / ((act_bias * (layer_weight + gamma * pos_layer_weight)).sum(axis=1))
            )
            * relevance
        ).sum(axis=1)
        return relevance


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


def isequal_sig_dict(signature1, signature2):
    if signature1.keys() != signature2.keys():
        return False
    keys = signature1.keys()
    return all(np.array_equal(signature1[key], signature2[key]) for key in keys)

def signature_dist(signature1, signature2):
    if signature1.keys() != signature2.keys():
        print('Cannot compare these signatures')
        return None
    distance = 0
    for key in signature1.keys():
        distance += int(abs(signature1[key] - signature2[key]).sum())
    return distance

