import os
from box import Box

from src.datasets.gaussianClouds import gaussianClouds
from src.datasets.uniformClouds import uniformClouds
from src.algorithms.autoencoder import autoencoder
from src.evaluations.parallelQualplots import parallelQualplots
from src.evaluations.linSubfctBarplots import linSubfctBarplots
from src.evaluations.linSub_unifPoints import linSub_unifPoints
from src.evaluations.subfunc_distmat import subfunc_distmat
from src.evaluations.linsubfct_parallelPlots import linsubfct_parallelPlots
from src.evaluations.singularValuePlots import singularValuePlots
from src.evaluations.evaluation import evaluation
from src.utils.config import config


def main():
    cfgs = load_cfgs()
    for cfg in cfgs:
        check_cfg_consistency(cfg)
        dataset, algorithm, evals = load_objects(cfg)
        if "train" in cfg.mode:
            algorithm.fit(dataset.data())
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


def load_objects(cfg):
    dataset = load_dataset(cfg)
    algorithm = load_algorithm(cfg)
    evals = load_evals(cfg)
    return dataset, algorithm, evals


def load_dataset(cfg):
    if cfg.dataset == "uniformClouds":
        dataset = uniformClouds(
            spacedim=cfg.datasets.uniformClouds.spacedim,
            clouddim=cfg.datasets.uniformClouds.clouddim,
            num_clouds=cfg.datasets.uniformClouds.num_clouds,
            num_datapoints=cfg.datasets.uniformClouds.num_datapoints,
            noise=cfg.datasets.uniformClouds.noise,
        )
    elif cfg.dataset == "gaussianClouds":
        dataset = gaussianClouds(
            spacedim=cfg.datasets.uniformClouds.spacedim,
            clouddim=cfg.datasets.uniformClouds.clouddim,
            num_clouds=cfg.datasets.uniformClouds.num_clouds,
            num_datapoints=cfg.datasets.uniformClouds.num_datapoints,
        )
    else:
        raise Exception("Could not create dataset.")
    return dataset


def load_algorithm(cfg):
    if "/" in cfg.algorithm:
        algorithm = load_algorithm_from_file(cfg.algorithm)
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
    return evals


if __name__ == "__main__":
    main()
