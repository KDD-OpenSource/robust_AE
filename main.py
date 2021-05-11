import os
from box import Box

from src.datasets.gaussianClouds import gaussianClouds
from src.datasets.uniformClouds import uniformClouds
from src.datasets.moons_2d import moons_2d
from src.datasets.parabola import parabola
from src.algorithms.autoencoder import autoencoder
from src.evaluations.parallelQualplots import parallelQualplots
from src.evaluations.linSubfctBarplots import linSubfctBarplots
from src.evaluations.linSub_unifPoints import linSub_unifPoints
from src.evaluations.subfunc_distmat import subfunc_distmat
from src.evaluations.linsubfct_parallelPlots import linsubfct_parallelPlots
from src.evaluations.singularValuePlots import singularValuePlots
from src.evaluations.closest_linsubfct_plot import closest_linsubfct_plot
from src.evaluations.boundary_2d_plot import boundary_2d_plot
from src.evaluations.inst_area_2d_plot import inst_area_2d_plot
from src.evaluations.border_dist_2d import border_dist_2d
from src.evaluations.border_dist_sort_plot import border_dist_sort_plot
from src.evaluations.evaluation import evaluation
from src.utils.config import config


def main():
    cfgs = load_cfgs()
    for cfg in cfgs:
        check_cfg_consistency(cfg)
        dataset, algorithm, eval_inst, evals = load_objects_cfgs(cfg)
        if "train" in cfg.mode:
            algorithm.fit(dataset.data())
            algorithm.save(eval_inst.get_run_folder())
        if "test" in cfg.mode:
            for evaluation in evals:
                evaluation.evaluate(dataset, algorithm)


def load_cfgs():
    cfgs = []
    cfgs.append(
        Box(config(os.path.join(os.getcwd(), "./configs/config_1.yaml")).config_dict)
    )
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
            spacedim=cfg.datasets.uniformClouds.spacedim,
            clouddim=cfg.datasets.uniformClouds.clouddim,
            num_clouds=cfg.datasets.uniformClouds.num_clouds,
            num_samples=cfg.datasets.uniformClouds.num_samples,
            num_anomalies=cfg.datasets.uniformClouds.num_anomalies,
            noise=cfg.datasets.uniformClouds.noise,
        )
    elif cfg.dataset == "gaussianClouds":
        dataset = gaussianClouds(
            spacedim=cfg.datasets.gaussianClouds.spacedim,
            clouddim=cfg.datasets.gaussianClouds.clouddim,
            num_clouds=cfg.datasets.gaussianClouds.num_clouds,
            num_samples=cfg.datasets.gaussianClouds.num_samples,
            num_anomalies=cfg.datasets.gaussianClouds.num_anomalies,
        )
    elif cfg.dataset == "moons_2d":
        dataset = moons_2d(
            num_samples=cfg.datasets.moons_2d.num_samples,
            num_anomalies=cfg.datasets.moons_2d.num_anomalies,
            noise=cfg.datasets.moons_2d.noise,
        )
    elif cfg.dataset == "parabola":
        dataset = parabola(
            num_samples=cfg.datasets.parabola.num_samples,
            num_anomalies=cfg.datasets.parabola.num_anomalies,
            noise=cfg.datasets.parabola.noise,
            spacedim=cfg.datasets.parabola.spacedim,
        )
    else:
        raise Exception("Could not create dataset.")
    return dataset


def load_algorithm(cfg):
    if "/" in cfg.algorithm:
        if "autoencoder" in cfg.algorithm:
            algorithm = autoencoder(topology = [2,1,2])
            algorithm.load(cfg.algorithm)
    elif cfg.algorithm == "autoencoder":
        algorithm = autoencoder(
            topology=cfg.algorithms.autoencoder.topology,
            num_epochs=cfg.algorithms.num_epochs,
            lr=cfg.algorithms.lr,
            collect_subfcts=cfg.algorithms.collect_subfcts,
            dropout=cfg.algorithms.autoencoder.dropout,
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
    return eval_inst, evals


if __name__ == "__main__":
    main()
