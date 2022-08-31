"""Util file for main"""
from src.utils.imports_ref import *
import sys
import os

def exec_cfg(cfg, start_timestamp):
    cur_time_str = time.strftime("%Y-%m-%dT%H:%M:%S")
    if cfg.repeat_experiments > 1:
        base_folder = cur_time_str
    else:
        base_folder = None
    if cfg.multiple_models != None:
        base_folder = (
            cur_time_str
            + "/"
            + cfg.ctx[: cfg.ctx.find("2021")]
            + cfg.multiple_models[cfg.multiple_models.rfind("/") + 1 :]
        )

    for repetition in range(cfg.repeat_experiments):
        if cfg.repeat_experiments > 1:
            dataset, algorithm, eval_inst, evals = load_objects_cfgs(
                cfg, base_folder=base_folder, exp_run=str(repetition)
            )
        else:
            dataset, algorithm, eval_inst, evals = load_objects_cfgs(
                cfg, base_folder=base_folder
            )

        if "train" in cfg.mode:
            init_logging(eval_inst.get_run_folder())
            logger = logging.getLogger(__name__)
            algorithm.fit(dataset, eval_inst.get_run_folder(), logger)
            algorithm.save(eval_inst.get_run_folder())
            dataset.save(os.path.join(eval_inst.get_run_folder(), "dataset"))

            dataset.save(
                os.path.join(
                    "./models/trained_models/", algorithm.name, "subfolder/dataset"
                )
            )
        if "opt_param" in cfg.mode:
            best_param, best_mean, best_gaussian, best_var, result_df = param_cross_val(
                cfg
            )
            res_dict = {}
            res_dict["param"] = cfg.algorithms.validation.parameter
            res_dict["value"] = best_param
            res_dict["error_mean"] = best_mean.astype(np.float64)
            res_dict["gaussian_mean"] = best_gaussian.astype(np.float64)
            res_dict["gaussian_var"] = best_gaussian.astype(np.float64)
            eval_inst.save_json(res_dict, "parameter_opt")
            eval_inst.save_csv(result_df, "parameter_all")
        if "test" in cfg.mode:
            for evaluation in evals:
                evaluation.evaluate(dataset, algorithm)
    cfg.to_json(filename=os.path.join(eval_inst.run_folder, "cfg.json"))
    print(f"Config {cfg.ctx} is done")

def load_cfgs():
    cfgs = []
    for arg in sys.argv[1:]:
        if arg[-1] == "/":
            for cfg in os.listdir(arg):
                if cfg[-4:] == "yaml":
                    cfgs.extend(read_cfg(arg + cfg))
        elif arg[-4:] == "yaml":
            cfgs.extend(read_cfg(arg))
        else:
            raise Exception("could not read argument")
    return cfgs


def read_cfg(cfg):
    cfgs = []
    cfgs.append(Box(config(os.path.join(os.getcwd(), cfg)).config_dict))
    if cfgs[-1].multiple_models is not None:
        #subfolder_blocklist = ['remaining_models']
        model_containing_folder = cfgs[-1].multiple_models
        model_list = os.listdir(model_containing_folder)
#        for blocked in subfolder_blocklist:
#            try:
#                model_list.remove(blocked)
#            except:
#                pass
#
#        import pdb; pdb.set_trace()
        for _ in range(len(model_list) - 1):
            cfgs.append(copy.deepcopy(cfgs[0]))
        for cfg, model_folder in zip(cfgs, model_list):
            model_path = model_containing_folder + "/" + model_folder
            cfg.algorithm = model_path
            cfg.ctx = cfg.ctx + "_" + model_path[model_path.rfind("/") + 1 :]
            dataset_path = model_path + "/dataset"
            try:
                import pdb; pdb.set_trace()
                data_properties = list(
                    filter(lambda x: "Properties" in x, os.listdir(dataset_path))
                )[0]
                with open(os.path.join(dataset_path, data_properties)) as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if row[0] == "name":
                            cfg.dataset = row[1]
                dataset_type = cfg.dataset
                for cfg_dataset in cfg.datasets.items():
                    if cfg_dataset[0] in dataset_type:
                        cfg_dataset[1].file_path = dataset_path
            except:
                pass
    return cfgs


def load_objects_cfgs(cfg, base_folder, exp_run=None):
    try:
        dataset = load_dataset(cfg)
    except:
        dataset = None
    try:
        algorithm = load_algorithm(cfg)
    except:
        algorithm = None
    try:
        eval_inst, evals = load_evals(cfg, base_folder, exp_run)
    except:
        eval_inst, evals = None, None
    return dataset, algorithm, eval_inst, evals


def load_dataset(cfg):
    if cfg.dataset == "sineNoise":
        dataset = sineNoise(
            file_path=cfg.datasets.sineNoise.file_path,
            subsample=cfg.datasets.subsample,
            scale=False,
            spacedim=cfg.datasets.sineNoise.spacedim,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.sineNoise.num_anomalies,
            num_testpoints=cfg.datasets.synthetic_test_samples,
        )
    elif cfg.dataset == "mnist":
        dataset = mnist(
            file_path=cfg.datasets.mnist.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.mnist.num_anomalies,
            normal_class=cfg.datasets.mnist.normal_class,
        )
    elif cfg.dataset == "wbc":
        dataset = wbc(
            file_path=cfg.datasets.wbc.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.wbc.num_anomalies,
        )
    elif cfg.dataset == "steel_plates_fault":
        dataset = steel_plates_fault(
            file_path=cfg.datasets.steel_plates_fault.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.steel_plates_fault.num_anomalies,
        )
    elif cfg.dataset == "qsar_biodeg":
        dataset = qsar_biodeg(
            file_path=cfg.datasets.qsar_biodeg.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.qsar_biodeg.num_anomalies,
        )
    elif cfg.dataset == "page_blocks":
        dataset = page_blocks(
            file_path=cfg.datasets.page_blocks.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.page_blocks.num_anomalies,
        )
    elif cfg.dataset == "gas_drift":
        dataset = gas_drift(
            file_path=cfg.datasets.gas_drift.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.gas_drift.num_anomalies,
        )
    elif cfg.dataset == "har":
        dataset = har(
            file_path=cfg.datasets.har.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.har.num_anomalies,
        )
    elif cfg.dataset == "satellite":
        dataset = satellite(
            file_path=cfg.datasets.satellite.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.satellite.num_anomalies,
        )
    elif cfg.dataset == "segment":
        dataset = segment(
            file_path=cfg.datasets.segment.file_path,
            subsample=cfg.datasets.subsample,
            scale=cfg.datasets.scale,
            num_samples=cfg.datasets.num_samples,
            num_anomalies=cfg.datasets.segment.num_anomalies,
        )
    elif cfg.dataset == "moteStrain":
        dataset = moteStrain(
            file_path=cfg.datasets.moteStrain.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "proximalPhalanxOutlineCorrect":
        dataset = proximalPhalanxOutlineCorrect(
            file_path=cfg.datasets.proximalPhalanxOutlineCorrect.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "phalangesOutlinesCorrect":
        dataset = phalangesOutlinesCorrect(
            file_path=cfg.datasets.phalangesOutlinesCorrect.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "twoLeadEcg":
        dataset = twoLeadEcg(
            file_path=cfg.datasets.twoLeadEcg.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "sonyAIBORobotSurface1":
        dataset = sonyAIBORobotSurface1(
            file_path=cfg.datasets.sonyAIBORobotSurface1.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "sonyAIBORobotSurface2":
        dataset = sonyAIBORobotSurface2(
            file_path=cfg.datasets.sonyAIBORobotSurface2.file_path,
            subsample=cfg.datasets.subsample,
        )
    else:
        raise Exception("Could not create dataset.")
    return dataset


def load_algorithm(cfg):
    if "/" in cfg.algorithm:
        # means that it is a path to an already trained one
        if "autoencoder" in cfg.algorithm:
            algorithm = autoencoder(
                topology=[2, 1, 2],
                fct_dist=cfg.algorithms.autoencoder.fct_dist,
                border_dist=cfg.algorithms.autoencoder.border_dist,
                num_border_points=cfg.algorithms.autoencoder.num_border_points,
            )
            algorithm.load(cfg.algorithm)
    else:
        if cfg.algorithm == "autoencoder":
            algorithm = autoencoder(
                train_robust_ae=cfg.algorithms.autoencoder.train_robust_ae,
                denoising=cfg.algorithms.autoencoder.denoising,
                fct_dist=cfg.algorithms.autoencoder.fct_dist,
                fct_dist_layer=cfg.algorithms.autoencoder.fct_dist_layer,
                border_dist=cfg.algorithms.autoencoder.border_dist,
                cross_point_sampling=cfg.algorithms.autoencoder.cross_point_sampling,
                lambda_border=cfg.algorithms.autoencoder.lambda_border,
                lambda_fct=cfg.algorithms.autoencoder.lambda_fct,
                topology=cfg.algorithms.autoencoder.topology,
                num_epochs=cfg.algorithms.num_epochs,
                dynamic_epochs=cfg.algorithms.dynamic_epochs,
                lr=cfg.algorithms.lr,
                collect_subfcts=cfg.algorithms.collect_subfcts,
                bias_shift=cfg.algorithms.autoencoder.bias_shift,
                push_factor=cfg.algorithms.push_factor,
                dropout=cfg.algorithms.autoencoder.dropout,
                L2Reg=cfg.algorithms.autoencoder.L2Reg,
                num_border_points=cfg.algorithms.autoencoder.num_border_points,
                save_interm_models=cfg.algorithms.save_interm_models,
            )
        else:
            raise Exception("Could not load algorithm")
    return algorithm


def load_evals(cfg, base_folder=None, exp_run=None):
    eval_inst = evaluation(base_folder)
    eval_inst.make_run_folder(ctx=cfg.ctx, exp_run=exp_run)
    evals = []
    if "marabou_ens_normal_rob" in cfg.evaluations:
        evals.append(marabou_ens_normal_rob(eval_inst=eval_inst, cfg=cfg))
    if "marabou_svdd_normal_rob" in cfg.evaluations:
        evals.append(marabou_svdd_normal_rob(eval_inst=eval_inst, cfg=cfg))
    if "marabou_ens_normal_rob_ae" in cfg.evaluations:
        evals.append(marabou_ens_normal_rob_ae(eval_inst=eval_inst, cfg=cfg))
    if "marabou_ens_normal_rob_submodels" in cfg.evaluations:
        evals.append(marabou_ens_normal_rob_submodels(eval_inst=eval_inst, cfg=cfg))
    if "marabou_robust" in cfg.evaluations:
        evals.append(marabou_robust(eval_inst=eval_inst))
    if "marabou_largest_error" in cfg.evaluations:
        evals.append(marabou_largest_error(eval_inst=eval_inst))
    if "inst_area_2d_plot" in cfg.evaluations:
        evals.append(inst_area_2d_plot(eval_inst=eval_inst))
    if "downstream_naiveBayes" in cfg.evaluations:
        evals.append(downstream_naiveBayes(eval_inst=eval_inst))
    if "downstream_knn" in cfg.evaluations:
        evals.append(downstream_knn(eval_inst=eval_inst))
    if "downstream_rf" in cfg.evaluations:
        evals.append(downstream_rf(eval_inst=eval_inst))
    return eval_inst, evals

if __name__ == "__main__":
    main()
