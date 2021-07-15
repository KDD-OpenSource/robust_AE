import os
import csv
import copy
import sys

import torch
import logging
import onnx
from onnx2pytorch import ConvertModel
from pytorch2keras.converter import pytorch_to_keras

from box import Box

from src.datasets.gaussianClouds import gaussianClouds
from src.datasets.uniformClouds import uniformClouds
from src.datasets.moons_2d import moons_2d
from src.datasets.parabola import parabola
from src.datasets.mnist import mnist
from src.datasets.creditcardFraud import creditcardFraud
from src.datasets.predictiveMaintenance import predictiveMaintenance
from src.algorithms.autoencoder import autoencoder
from src.evaluations.parallelQualplots import parallelQualplots
from src.evaluations.tsne_latent import tsne_latent
from src.evaluations.linSubfctBarplots import linSubfctBarplots
from src.evaluations.linSub_unifPoints import linSub_unifPoints
from src.evaluations.subfunc_distmat import subfunc_distmat
from src.evaluations.linsubfct_parallelPlots import linsubfct_parallelPlots
from src.evaluations.linsubfct_distr import linsubfct_distr
from src.evaluations.singularValuePlots import singularValuePlots
from src.evaluations.closest_linsubfct_plot import closest_linsubfct_plot
from src.evaluations.boundary_2d_plot import boundary_2d_plot
from src.evaluations.inst_area_2d_plot import inst_area_2d_plot
from src.evaluations.border_dist_2d import border_dist_2d
from src.evaluations.border_dist_sort_plot import border_dist_sort_plot
from src.evaluations.error_border_dist_plot import error_border_dist_plot
from src.evaluations.orig_latent_dist_ratios import orig_latent_dist_ratios
from src.evaluations.error_label_mean_dist_plot import (
    error_label_mean_dist_plot)
from src.evaluations.error_border_dist_plot_colored import (
    error_border_dist_plot_colored)
from src.evaluations.error_border_dist_plot_anomalies import (
    error_border_dist_plot_anomalies)
from src.evaluations.plot_mnist_samples import plot_mnist_samples
from src.evaluations.image_lrp import image_lrp
from src.evaluations.image_feature_imp import image_feature_imp
from src.evaluations.qual_by_border_dist_plot import qual_by_border_dist_plot
from src.evaluations.marabou_test import marabou_test
from src.evaluations.fct_change_by_border_dist_qual import (
    fct_change_by_border_dist_qual)
from src.evaluations.bias_feature_imp import bias_feature_imp
from src.evaluations.interpolation_func_diffs_pairs import (
    interpolation_func_diffs_pairs)
from src.evaluations.mnist_interpolation_func_diffs_pairs import (
        mnist_interpolation_func_diffs_pairs)
from src.evaluations.evaluation import evaluation
from src.utils.config import config, init_logging


def main():
    cfgs = load_cfgs()
    for cfg in cfgs:
        check_cfg_consistency(cfg)
        dataset, algorithm, eval_inst, evals = load_objects_cfgs(cfg)
        if "train" in cfg.mode:
            init_logging(eval_inst.get_run_folder())
            logger = logging.getLogger(__name__)
            algorithm.fit(dataset.data(), eval_inst.get_run_folder(), logger)
            algorithm.save(eval_inst.get_run_folder())
            dataset.save(os.path.join(
                eval_inst.get_run_folder(),
                'dataset'
                ))

            dataset.save(os.path.join(
                './models/trained_models/',
                algorithm.name,
                'dataset',
                dataset.name
                ))
        if "test" in cfg.mode:
            for evaluation in evals:
                evaluation.evaluate(dataset, algorithm)


def load_cfgs():
    cfgs = []
    for arg in sys.argv[1:]:
        if arg[-1] == '/':
            for cfg in os.listdir(arg):
                if cfg[-4:] == 'yaml':
                    cfgs.extend(read_cfg(arg + cfg))
        elif arg[-4:] == 'yaml':
            cfgs.extend(read_cfg(arg))
        else:
            raise Exception('could not read argument')
    return cfgs

def read_cfg(cfg):
    cfgs = []
    cfgs.append(
        Box(config(os.path.join(os.getcwd(), cfg)).config_dict)
    )
    if cfgs[-1].multiple_models is not None:
        model_containing_folder = cfgs[-1].multiple_models
        for _ in range(len(os.listdir(model_containing_folder))-1):
            cfgs.append(copy.deepcopy(cfgs[0]))
        for cfg, model_folder in zip(
                cfgs,os.listdir(model_containing_folder)):
            model_path = model_containing_folder +'/'+ model_folder
            cfg.algorithm = model_path
            cfg.ctx = cfg.ctx + '_' + model_path[model_path.rfind('/')+1:]
            dataset_path = model_path + '/dataset'
            data_properties = list(filter(
                lambda x: 'Properties' in x, os.listdir(dataset_path)))[0]
            with open(os.path.join(dataset_path, data_properties)) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if row[0] == 'name':
                        cfg.dataset = row[1]
            dataset_type = cfg.dataset
            for cfg_dataset in cfg.datasets.items():
                if cfg_dataset[0] == dataset_type:
                    cfg_dataset[1].file_path = dataset_path
    return cfgs


def check_cfg_consistency(cfg):
    pass
    # check data dim = input dim
    # check if for each evaluation linsubfctcount is required
    # if no train, then you must have a load field


def load_objects_cfgs(cfg):
    dataset = load_dataset(cfg)
    algorithm = load_algorithm(cfg)
    eval_inst, evals = load_evals(cfg)
    return dataset, algorithm, eval_inst, evals


def load_dataset(cfg):
    if cfg.dataset == "uniformClouds":
        dataset = uniformClouds(
            file_path=cfg.datasets.uniformClouds.file_path,
            subsample=cfg.datasets.subsample,
            spacedim=cfg.datasets.uniformClouds.spacedim,
            clouddim=cfg.datasets.uniformClouds.clouddim,
            num_clouds=cfg.datasets.uniformClouds.num_clouds,
            num_samples=cfg.datasets.uniformClouds.num_samples,
            num_anomalies=cfg.datasets.uniformClouds.num_anomalies,
            noise=cfg.datasets.uniformClouds.noise,
        )
    elif cfg.dataset == "gaussianClouds":
        dataset = gaussianClouds(
            file_path=cfg.datasets.gaussianClouds.file_path,
            subsample=cfg.datasets.subsample,
            spacedim=cfg.datasets.gaussianClouds.spacedim,
            clouddim=cfg.datasets.gaussianClouds.clouddim,
            num_clouds=cfg.datasets.gaussianClouds.num_clouds,
            num_samples=cfg.datasets.gaussianClouds.num_samples,
            num_anomalies=cfg.datasets.gaussianClouds.num_anomalies,
        )
    elif cfg.dataset == "moons_2d":
        dataset = moons_2d(
            file_path=cfg.datasets.moons_2d.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.moons_2d.num_samples,
            num_anomalies=cfg.datasets.moons_2d.num_anomalies,
            noise=cfg.datasets.moons_2d.noise,
        )
    elif cfg.dataset == "parabola":
        dataset = parabola(
            file_path=cfg.datasets.parabola.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.parabola.num_samples,
            num_anomalies=cfg.datasets.parabola.num_anomalies,
            noise=cfg.datasets.parabola.noise,
            spacedim=cfg.datasets.parabola.spacedim,
        )
    elif cfg.dataset == "mnist":
        dataset = mnist(
            file_path=cfg.datasets.mnist.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.mnist.num_samples,
            num_anomalies=cfg.datasets.mnist.num_anomalies,
        )
    elif cfg.dataset == "creditcardFraud":
        dataset = creditcardFraud(
            file_path=cfg.datasets.creditcardFraud.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.creditcardFraud.num_samples,
            num_anomalies=cfg.datasets.creditcardFraud.num_anomalies,
        )
    elif cfg.dataset == "predictiveMaintenance":
        dataset = predictiveMaintenance(
            file_path=cfg.datasets.predictiveMaintenance.file_path,
            subsample=cfg.datasets.subsample,
            num_samples=cfg.datasets.predictiveMaintenance.num_samples,
            num_anomalies=cfg.datasets.predictiveMaintenance.num_anomalies,
            window_size=cfg.datasets.predictiveMaintenance.window_size,
        )
    else:
        raise Exception("Could not create dataset.")
    return dataset


def load_algorithm(cfg):
    if "/" in cfg.algorithm:
        if "autoencoder" in cfg.algorithm:
            algorithm = autoencoder(
                    topology = [2,1,2],
                    fct_dist = cfg.algorithms.autoencoder.fct_dist,
                    border_dist = cfg.algorithms.autoencoder.border_dist,
                    num_border_points=cfg.algorithms.autoencoder.num_border_points)
            algorithm.load(cfg.algorithm)
    elif cfg.algorithm == "autoencoder":
        algorithm = autoencoder(
            border_dist=cfg.algorithms.autoencoder.border_dist,
            fct_dist=cfg.algorithms.autoencoder.fct_dist,
            lambda_border=cfg.algorithms.autoencoder.lambda_border,
            lambda_fct=cfg.algorithms.autoencoder.lambda_fct,
            topology=cfg.algorithms.autoencoder.topology,
            num_epochs=cfg.algorithms.num_epochs,
            dynamic_epochs=cfg.algorithms.dynamic_epochs,
            lr=cfg.algorithms.lr,
            collect_subfcts=cfg.algorithms.collect_subfcts,
            dropout=cfg.algorithms.autoencoder.dropout,
            L2Reg=cfg.algorithms.autoencoder.L2Reg,
            num_border_points=cfg.algorithms.autoencoder.num_border_points,
        )
    else:
        raise Exception("Could not load algorithm")
    return algorithm


def load_evals(cfg):
    eval_inst = evaluation()
    eval_inst.make_run_folder(ctx=cfg.ctx)
    evals = []
    if "parallelQualplots" in cfg.evaluations:
        evals.append(parallelQualplots(eval_inst=eval_inst))
    if "linSubfctBarplots" in cfg.evaluations:
        evals.append(linSubfctBarplots(eval_inst=eval_inst))
    if "linSub_unifPoints" in cfg.evaluations:
        evals.append(linSub_unifPoints(eval_inst=eval_inst))
    if "subfunc_distmat" in cfg.evaluations:
        evals.append(subfunc_distmat(eval_inst=eval_inst))
    if "linsubfct_parallelPlots" in cfg.evaluations:
        evals.append(linsubfct_parallelPlots(eval_inst=eval_inst))
    if "linsubfct_distr" in cfg.evaluations:
        evals.append(linsubfct_distr(eval_inst=eval_inst))
    if "singularValuePlots" in cfg.evaluations:
        evals.append(singularValuePlots(eval_inst=eval_inst))
    if "closest_linsubfct_plot" in cfg.evaluations:
        evals.append(closest_linsubfct_plot(eval_inst=eval_inst))
    if "boundary_2d_plot" in cfg.evaluations:
        evals.append(boundary_2d_plot(eval_inst=eval_inst))
    if "inst_area_2d_plot" in cfg.evaluations:
        evals.append(inst_area_2d_plot(eval_inst=eval_inst))
    if "border_dist_2d" in cfg.evaluations:
        evals.append(border_dist_2d(eval_inst=eval_inst))
    if "border_dist_sort_plot" in cfg.evaluations:
        evals.append(border_dist_sort_plot(eval_inst=eval_inst))
    if "error_border_dist_plot" in cfg.evaluations:
        evals.append(error_border_dist_plot(eval_inst=eval_inst))
    if "error_label_mean_dist_plot" in cfg.evaluations:
        evals.append(error_label_mean_dist_plot(eval_inst=eval_inst))
    if "error_border_dist_plot_colored" in cfg.evaluations:
        evals.append(error_border_dist_plot_colored(eval_inst=eval_inst))
    if "error_border_dist_plot_anomalies" in cfg.evaluations:
        evals.append(error_border_dist_plot_anomalies(eval_inst=eval_inst))
    if "plot_mnist_samples" in cfg.evaluations:
        evals.append(plot_mnist_samples(eval_inst=eval_inst))
    if "image_lrp" in cfg.evaluations:
        evals.append(image_lrp(eval_inst=eval_inst))
    if "image_feature_imp" in cfg.evaluations:
        evals.append(image_feature_imp(eval_inst=eval_inst))
    if "qual_by_border_dist_plot" in cfg.evaluations:
        evals.append(qual_by_border_dist_plot(eval_inst=eval_inst))
    if "fct_change_by_border_dist_qual" in cfg.evaluations:
        evals.append(fct_change_by_border_dist_qual(eval_inst=eval_inst))
    if "marabou_test" in cfg.evaluations:
        evals.append(marabou_test(eval_inst=eval_inst))
    if "bias_feature_imp" in cfg.evaluations:
        evals.append(bias_feature_imp(eval_inst=eval_inst))
    if "interpolation_func_diffs_pairs" in cfg.evaluations:
        evals.append(interpolation_func_diffs_pairs(eval_inst=eval_inst))
    if "mnist_interpolation_func_diffs_pairs" in cfg.evaluations:
        evals.append(mnist_interpolation_func_diffs_pairs(eval_inst=eval_inst))
    if "tsne_latent" in cfg.evaluations:
        evals.append(tsne_latent(eval_inst=eval_inst))
    if "orig_latent_dist_ratios" in cfg.evaluations:
        evals.append(orig_latent_dist_ratios(eval_inst=eval_inst))
    return eval_inst, evals


if __name__ == "__main__":
    main()
