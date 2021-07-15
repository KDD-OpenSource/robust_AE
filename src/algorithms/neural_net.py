from pprint import pprint

import abc
import gc
import math
import copy
import hashlib
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from functools import reduce

class neural_net:
    def __init__(
        self,
        name: str,
        num_epochs: int = 100,
        dynamic_epochs: bool = False,
        batch_size: int = 20,
        lr: float = 1e-3,
        seed: int = None,
    ):
        self.name = name
        self.num_epochs = num_epochs
        self.dynamic_epochs = dynamic_epochs
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

    def get_data_loader(self, X: pd.DataFrame):
        if self.dynamic_epochs:
            data_sampler = torch.utils.data.RandomSampler(data_source=X,
                    replacement = True, num_samples = 500)
            self.num_epochs = self.num_epochs * int(X.shape[0]/500)

            data_loader = DataLoader(
                dataset=X.values,
                batch_size=self.batch_size,
                drop_last=True,
                pin_memory=True,
                sampler = data_sampler
            )
        else:
            data_loader = DataLoader(
                dataset=X.values,
                batch_size=self.batch_size,
                drop_last=True,
                pin_memory=True,
                )
        return data_loader




    def get_lin_subfct(self, neural_net_mod, instance, max_layer=None):
#        if max_layer is not None:
#            neural_net = copy.deepcopy(neural_net_mod.get_neural_net()[:max_layer])
#        else:
#            neural_net = copy.deepcopy(neural_net_mod.get_neural_net())
#        _, intermed_results = neural_net_mod(instance)
#        linear_layers_list = []
#        bias_list = []
#        for key, value in intermed_results.items():
#            if (
#                int(key) < len(neural_net) - 1
#                and isinstance(neural_net[int(key)], nn.Linear)
#                and isinstance(neural_net[int(key) + 1], nn.ReLU)
#            ):
#                reluKey = str(int(key) + 1)
#                matrix = neural_net[int(key)].weight.detach().numpy()
#                bias = neural_net[int(key)].bias.detach().numpy()
#                for value_ind in range(len(intermed_results[reluKey])):
#                    if intermed_results[reluKey][value_ind] == 0:
#                        matrix[value_ind] = 0
#                        bias[value_ind] = 0
#                linear_layers_list.append(matrix)
#                bias_list.append(bias)
#
#            # go through all layers
#        if isinstance(neural_net[-1], nn.Linear):
#            linear_layers_list.append(neural_net[-1].weight.detach().numpy())
#            bias_list.append(neural_net[-1].bias.detach().numpy())
#        linear_layers_transposed = list(
#            map(lambda x: x.transpose(), linear_layers_list)
#        )
#        result_matrix = reduce(np.matmul, linear_layers_transposed)
#
#        result_bias = bias_list[-1]
#        for bias_ind in range(len(bias_list) - 2, -1, -1):
#            bias_matrix = reduce(np.matmul, linear_layers_transposed[bias_ind + 1 :])
#            result_bias += np.matmul(bias_list[bias_ind], bias_matrix)
#        result_matrix = result_matrix.transpose()
#        signature = self.get_signature(neural_net_mod, instance)
#        lin_subfct = linearSubfunction(result_matrix, result_bias, signature)

        forward_help_fct = smallest_k_dist_loss(1)
        neural_net = copy.deepcopy(neural_net_mod.get_neural_net())
        mat, bias, relus = forward_help_fct.calculate_inst(neural_net,
                instance, max_layer=max_layer)

        nn_layers = list(neural_net_mod.get_neural_net())
        relu_key_list = []
        for layer_ind in range(len(nn_layers)):
            if isinstance(nn_layers[layer_ind], nn.ReLU):
                relu_key_list.append(str(layer_ind))
        relu_dict = {}
        for layer_ind, relu_vals in zip(relu_key_list, relus):
            relu_dict[layer_ind] = relu_vals.numpy().flatten()
        lin_subfct = linearSubfunction(mat.detach().numpy(),
                bias.detach().numpy(), relu_dict)

        return lin_subfct

#    def calculate_inst(self, neural_net, instance, dists: bool=False):
#        _, intermed_results = neural_net(instance)
#        relus, weights, biases = self.get_neural_net_info(neural_net, intermed_results)
#
#        if dists:
#            distances = torch.tensor([])
#            cross_points = torch.tensor([])
#
#        V = torch.eye(instance.shape[0])
#        a = torch.zeros(instance.shape[0])
#
#        for ind in range(0, len(weights)):
#            V = torch.matmul(weights[ind], V)
#            a = (biases[ind] + torch.matmul(weights[ind] , a))
#            if dists:
#                intermed_res_ind = int(ind * 2)
#                dist = (torch.matmul(V, instance)+ a)/torch.norm(V, dim=1)
#                normals = V.transpose(0,1)/torch.norm(V, dim=1)
#                dist_normals = normals * 1.05 * dist
#                cross_point = instance - torch.transpose(dist_normals,0,1)
#                cross_points = torch.cat((cross_points, cross_point))
#                dist = abs(dist)
#                distances = torch.cat((distances, dist))
#            V = V * relus[ind]
#            a = a * relus[ind][:,0]
#
#
#        inst_mat = torch.matmul(neural_net[len(neural_net)-1].weight, V)
#        inst_bias = neural_net[len(neural_net)-1].bias + torch.matmul(neural_net[len(neural_net)-1].weight, a)
#
#        if dists:
#            return inst_mat, inst_bias, distances, cross_points
#        else:
#            return inst_mat, inst_bias, relus

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
            drop_last=False,
            pin_memory=True,
        )
        ctr = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                functions.append(self.get_lin_subfct(neural_net_mod,
                    inst.float()))
                ctr += 1
                #print(ctr)
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
            drop_last=False,
            pin_memory=True,
        )
        inst_func_pairs = []
        for inst_batch in data_loader:
            for inst in inst_batch:
                inst_func_pairs.append(
                    (inst.float(), self.get_lin_subfct(neural_net_mod,
                        inst.float()))
                )
        return inst_func_pairs

    def assign_bias_feature_imps(self, neural_net_mod, X):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        inst_imp_pairs = []
        for inst_batch in data_loader:
            for inst in inst_batch:
                feature_imp = self.apply_lin_func_without_bias(neural_net_mod,
                        inst)
                bias_imp = self.lin_func_bias_imp(neural_net_mod, inst)
                feature_imp_abs_sum = abs(feature_imp).sum()
                bias_imp_abs_sum = abs(bias_imp).sum()
                imp_sum = feature_imp_abs_sum + bias_imp_abs_sum
                inst_imp_pairs.append((inst.float(), imp_sum, feature_imp_abs_sum,
                        bias_imp_abs_sum))
        return inst_imp_pairs

    def assign_lin_subfcts_ind(self, neural_net_mod, X):
        functions = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        inst_func_pairs = []
        for inst_batch in data_loader:
            for inst in inst_batch:
                inst_func_pairs.append(
                    (inst.float(), self.get_lin_subfct(neural_net_mod,
                        inst.float()))
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

    #def assign_border_dists(self, neural_net_mod, X):
        #return assign_top_k_border_dists(neural_net_mod, X, 1)
#        neural_net_mod = copy.deepcopy(neural_net_mod)
#        data_loader = DataLoader(
#            dataset=X.values,
#            batch_size=self.batch_size,
#            drop_last=False,
#            pin_memory=True,
#        )
#        sample_dist_pairs = []
#        ctr = 0
#        for inst_batch in data_loader:
#            for inst in inst_batch:
#                closest_func_Boundary = self.get_closest_funcBoundary(
#                        neural_net_mod, inst.float())
#                if closest_func_Boundary is not None:
#                    sample_dist_pairs.append(
#                        (inst, closest_func_Boundary[1])
#                    )
#                ctr += 1
#                print(ctr)
#        return sample_dist_pairs

    def assign_top_k_border_dists(self, neural_net_mod, X, k):
        top_k_dists = []
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        sample_dist_pairs = []
        ctr = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                top_k_dists_sum = smallest_k_dist_loss(k,
                        border_dist=True,
                        fct_dist=False)(inst.unsqueeze(0),
                            neural_net_mod)[0].detach().numpy()
                #print(top_k_dists_sum)
                #top_k_dists_sum = self.get_top_k_dists_sum(
                        #neural_net_mod, inst.float(), k)
                if top_k_dists_sum is not None:
                    sample_dist_pairs.append(
                        (inst, top_k_dists_sum)
                    )
                #ctr += 1
                #print(ctr)
        return sample_dist_pairs

    def assign_border_dists(self, neural_net_mod, X):
        return self.assign_top_k_border_dists(neural_net_mod, X, 1)
#        neural_net_mod = copy.deepcopy(neural_net_mod)
#        data_loader = DataLoader(
#            dataset=X.values,
#            batch_size=self.batch_size,
#            drop_last=False,
#            pin_memory=True,
#        )
#        sample_dist_pairs = []
#        ctr = 0
#        for inst_batch in data_loader:
#            for inst in inst_batch:
#                closest_func_Boundary = self.get_closest_funcBoundary(
#                        neural_net_mod, inst.float())
#                if closest_func_Boundary is not None:
#                    sample_dist_pairs.append(
#                        (inst, closest_func_Boundary[1])
#                    )
#                ctr += 1
#                print(ctr)
#        return sample_dist_pairs

    def assign_most_far_border_dists(self, neural_net_mod, X):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        sample_dist_pairs = []
        ctr = 0
        for inst_batch in data_loader:
            for inst in inst_batch:
                closest_func_Boundary = self.get_most_far_funcBoundary(
                        neural_net_mod, inst.float())
                if closest_func_Boundary is not None:
                    sample_dist_pairs.append(
                        (inst.float(), closest_func_Boundary[1])
                    )
                ctr += 1
                print(ctr)
        return sample_dist_pairs

    def get_points_of_linsubfcts(self, neural_net_mod, X):
        functions = {}
        neural_net_mod = copy.deepcopy(neural_net_mod)
        data_loader = DataLoader(
            dataset=X.values,
            batch_size=self.batch_size,
            drop_last=False,
            pin_memory=True,
        )
        for inst_batch in data_loader:
            for inst in inst_batch:
                function = self.get_lin_subfct(neural_net_mod, inst.float())
                if function not in functions.keys():
                    functions[function] = []
                else:
                    functions[function].append(inst.float())
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
        try:
            cross_points_sorted = sorted(list(zip(dists, cross_points)),
                    key=lambda x:x[0])
        except:
            import pdb; pdb.set_trace()
        for dist_cross_point_pair in cross_points_sorted:
            if self.check_true_cross_point(neural_net_mod, instance,
                    dist_cross_point_pair[1]):
                return (dist_cross_point_pair[1],
            dist_cross_point_pair[0])

    def get_top_k_funcBoundDists(self, neural_net_mod, instance, k):
        neural_net_mod = copy.deepcopy(neural_net_mod)
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
        cross_points_sorted = sorted(list(zip(dists, cross_points)),
                key= lambda x:x[0])
        return cross_points_sorted[:k]

    def get_top_k_dists_sum(self, neural_net_mod, instance, k):
        cross_points = self.get_top_k_funcBoundDists(neural_net_mod, instance,
                k)
        dists = list(map(lambda x: x[0], cross_points))
        return sum(dists)

    def get_most_far_funcBoundary(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
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
        cross_points_sorted = sorted(list(zip(dists, cross_points)), reverse =
                True)
        for dist_cross_point_pair in cross_points_sorted:
            if self.check_true_cross_point(neural_net_mod, instance,
                    dist_cross_point_pair[1]):
                return (dist_cross_point_pair[1],
            dist_cross_point_pair[0])

    def get_closest_afterCross_fct(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
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
                cross_point = dist_cross_point_pair[1]
                after_cross = instance + 1.01 * (cross_point - instance)
                return (after_cross, self.get_lin_subfct(neural_net_mod, after_cross))

    def get_most_far_afterCross_fct(self, neural_net_mod, instance):
        neural_net_mod = copy.deepcopy(neural_net_mod)
        instance_sig = self.get_signature(neural_net_mod, instance)
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
        cross_points_sorted = sorted(list(zip(dists, cross_points)),
                reverse=True)
        for dist_cross_point_pair in cross_points_sorted:
            if self.check_true_cross_point(neural_net_mod, instance,
                    dist_cross_point_pair[1]):
                cross_point = dist_cross_point_pair[1]
                after_cross = instance + 1.01 * (cross_point - instance)
                return (after_cross, self.get_lin_subfct(neural_net_mod, after_cross))
    #def get_afterCross_fct_image(self, neural_net_mod, instance):



    #def apply_closest_different_fct(self, neural_net_mod, instance):
        #after_cross, _ = self.get_lin_subfct
        #pass


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
            #for most far only
            return True
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
        # neuron[0] is the weight_vector
        # neuron[1] is the bias
        if np.linalg.norm(neuron[0]) == 0:
            return None
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
    def lin_func_feature_imp(self, neural_net_mod, instance):
        lin_func = self.get_lin_subfct(neural_net_mod, instance)
        feature_imps = lin_func.matrix.sum(axis=0)
        return feature_imps

    def apply_lin_func_without_bias(self, neural_net_mod, instance):
        lin_func = self.get_lin_subfct(neural_net_mod, instance)
        result = np.matmul(lin_func.matrix, instance)
        return result


    def lin_func_bias_imp(self, neural_net_mod, instance):
        lin_func = self.get_lin_subfct(neural_net_mod, instance)
        bias_imps = lin_func.bias
        return bias_imps

    def lrp_ae(self, neural_net_mod, instance, gamma=1):
        input_relevance = np.zeros(instance.size())
        neural_net_mod = copy.deepcopy(neural_net_mod)
        output, intermed_res = neural_net_mod(instance)
        layer_inds = range(len(neural_net_mod.get_neural_net()))
        gamma = gamma
        error = nn.MSELoss()(instance, output)
        relevance = torch.tensor((np.array(instance) - output.detach().numpy()) ** 2)

        relevance_bias = 0
        for layer_ind in layer_inds[::-1]:
            layer = neural_net_mod.get_neural_net()[layer_ind]
            if layer_ind != 0:
                activation = intermed_res[str(layer_ind - 1)]
            else:
                activation = instance
            if isinstance(layer, nn.Linear):
                # first if: not last layer
                # second if: right layer for relu
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
        return relevance.detach().numpy(), relevance_bias

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
    def __init__(self, matrix, bias, signature):
        self.matrix = matrix
        self.bias = bias
        self.signature = signature
        self.name = self.__hash__()

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        #import pdb; pdb.set_trace()
        #return (self.matrix == other.matrix).all() and (self.bias == other.bias).all()
        return ((abs(self.matrix - other.matrix)).sum() + (abs(self.bias -
            other.bias)).sum()) < 1

    def __key(self):
        return (self.matrix.tobytes(), self.bias.tobytes())

    def __hash__(self):
        return hash(self.__key())

    def dist(self, other):
        return self.dist_bias(other) + self.dist_mat(other)

    def dist_bias(self, other):
        diff = self.bias - other.bias
        squared = diff ** 2
        summed = squared.sum()
        result = math.sqrt(summed)
        return result

    def dist_mat(self, other):
        diff = self.matrix - other.matrix
        scalar_prod = np.trace(np.matmul(diff, diff.transpose()))
        result = math.sqrt(scalar_prod)
        return result

    def dist_sign(self, other):
        return signature_dist(self.signature, other.signature)

    def dist_sign_weighted(self, other):
        if self.signature.keys() != other.signature.keys():
            print('Cannot compare these signatures')
            return None
        distance = 0
        for key in self.signature.keys():
            num_changes = int(abs(self.signature[key] -
                other.signature[key]).sum())
            # calculates /2 and round up  (mapping [1,3,5,7] to [1,2,3,4])
            #factor_inverse = 1/(int((int(key)/2)+1))
            factor = int((int(key)/2)+1)
            distance += factor*num_changes
        return distance

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

class smallest_k_dist_loss(nn.Module):
    def __init__(self, k, border_dist = False, penal_dist = None, fct_dist = False):
        self.k = k
        self.border_dist = border_dist
        self.penal_dist = penal_dist
        self.fct_dist = fct_dist
        super(smallest_k_dist_loss, self).__init__()

    def forward(self, inputs, module):
        border_dist_sum = 0
        fct_dist_sum = 0
        # iterate over batches
        #import pdb; pdb.set_trace()
        for inst in inputs:
            # code to be executed in following loop to determine nan-culprit:
            # module.zero_grad()
            # print(self.smallest_k_dists(module, inst).backward())
            # OR print(self.smallest_k_dists(module, inst)[1].backward())
            # OR print(self.smallest_k_dists(module, inst)[0].backward())
            # for elem in [0,2,4,6,...]:
            # print(module.get_neural_net()[elem].weight.grad.isnan().sum())
            border_dist, fct_dist = self.smallest_k_dists(module, inst)
            border_dist_sum += border_dist
            fct_dist_sum += fct_dist
        return border_dist_sum, fct_dist_sum

    def smallest_k_dists(self, neural_net_mod, instance):
        border_result = 0
        fct_result = 0
        if not self.border_dist and not self.fct_dist:
            return 0,0
        neural_net = neural_net_mod.get_neural_net()

        fct_dists = torch.tensor([])
        inst_mat, inst_bias, dists, cross_points = self.calculate_inst(
                neural_net, instance, dists=True)
        #gc.collect()
        if dists.isnan().sum() > 0:
            #import pdb; pdb.set_trace()
            #is the solution (probably)
            return 0,0
        if dists.isinf().sum() > 0:
            #import pdb; pdb.set_trace()
            # for debugging purposes
            #inst_mat, inst_bias, dists, cross_points = self.calculate_inst(
                    #neural_net, instance, dists=True)
            #is the solution (probably)
            return 0,0
        top_k_dists = torch.topk(input=dists, k=self.k,
                largest=False)
        if self.border_dist:
            # penalize every border that is closer than self.penal_dist (version from paper)
            if self.penal_dist:
                top_k_maxed = torch.max(torch.zeros(self.k),
                        1-(top_k_dists[0]/self.penal_dist))
            else:
                # simply return the (true) distance (for evaluation purposes)
                top_k_maxed = top_k_dists[0]

            # penelize every closest border (my version) (how to prevent
            # dividing by 0 here?)
            #top_k_maxed = 1/top_k_dists[0]
            border_result += top_k_maxed.sum()/self.k

        if self.fct_dist:
            for top_k_ind in top_k_dists[1]:
                cross_point = cross_points[top_k_ind]
                cross_point_mat, cross_point_bias, _ = self.calculate_inst(
                        neural_net, cross_point, dists = False)
                mat_dist = self.calc_mat_diff(inst_mat, cross_point_mat)
                bias_dist = self.calc_bias_diff(inst_bias, cross_point_bias)
                if mat_dist > 0 and bias_dist > 0:
                    fct_dist = (mat_dist + bias_dist).unsqueeze(0)
                else:
                    fct_dist = (mat_dist + bias_dist).unsqueeze(0)
                    fct_dist = fct_dist.detach()
                #fct_dist = self.calc_fct_diff(inst_mat, inst_bias, cross_point_mat,
                        #cross_point_bias)
                #if fct_dist < 1e-3:
                    #fct_dist = fct_dist.detach()

                fct_dists = torch.cat((fct_dists,fct_dist))
            fct_result += fct_dists.sum()/self.k
        return border_result, fct_result

    def calculate_inst(self, neural_net, instance, dists: bool=False, max_layer
            = None):
        if max_layer:
            neural_net = copy.deepcopy(neural_net[:max_layer])
        _, intermed_results = neural_net(instance)
        relus, weights, biases = self.get_neural_net_info(neural_net, intermed_results)

        if dists:
            distances = torch.tensor([])
            cross_points = torch.tensor([])

        V = torch.eye(instance.shape[0])
        a = torch.zeros(instance.shape[0])

        for ind in range(0, len(weights)):
            V = torch.matmul(weights[ind], V)
            a = (biases[ind] + torch.matmul(weights[ind] , a))
            if dists:
                intermed_res_ind = int(ind * 2)
                dist = (torch.matmul(V, instance)+ a)/torch.norm(V, dim=1)
                normals = V.transpose(0,1)/torch.norm(V, dim=1)
                dist_normals = normals * 1.05 * dist
                cross_point = instance - torch.transpose(dist_normals,0,1)
                cross_points = torch.cat((cross_points, cross_point))
                dist = abs(dist)
                distances = torch.cat((distances, dist))
            V = V * relus[ind]
            a = a * relus[ind][:,0]


        inst_mat = torch.matmul(neural_net[len(neural_net)-1].weight, V)
        inst_bias = neural_net[len(neural_net)-1].bias + torch.matmul(neural_net[len(neural_net)-1].weight, a)

        if dists:
            return inst_mat, inst_bias, distances, cross_points
        else:
            return inst_mat, inst_bias, relus

    def get_neural_net_info(self, neural_net, intermed_results):
        relus = []
        weights = []
        biases = []
        for key, value in intermed_results.items():
            if (
                int(key) < len(neural_net) - 1
                and isinstance(neural_net[int(key)], nn.Linear)
                and isinstance(neural_net[int(key) + 1], nn.ReLU)
            ):
                reluKey = str(int(key) + 1)
                relu = intermed_results[reluKey]
                relu = torch.unsqueeze(torch.greater(relu,
                        0).type(torch.FloatTensor), dim=1)
                relus.append(relu)
                weights.append(neural_net[int(key)].weight)
                biases.append(neural_net[int(key)].bias)
        return relus, weights, biases

    def calc_fct_diff(self, inst_mat, inst_bias, other_mat,
            other_bias):
        res_mat = self.calc_mat_diff(inst_mat, other_mat)
        res_bias = self.calc_bias_diff(inst_bias, other_bias)
        return (res_mat + res_bias).unsqueeze(0)

    def calc_mat_diff(self, inst_mat, other_mat):
        diff_mat = other_mat - inst_mat
        scalar_prod = torch.trace(torch.matmul(diff_mat,
            torch.transpose(diff_mat,0,1)))
        res_mat = torch.sqrt(scalar_prod)
        return res_mat

    def calc_bias_diff(self, inst_bias, other_bias):
        diff_bias = other_bias - inst_bias
        squared = diff_bias ** 2
        summed = squared.sum()
        res_bias = torch.sqrt(summed)
        return res_bias
