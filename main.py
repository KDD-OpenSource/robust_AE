from imports import *


def main():
    parallel = False
    cfgs = load_cfgs()
    start_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    if parallel:
        pool = mp.Pool(int(mp.cpu_count()))
        for cfg in cfgs:
            pool.apply_async(exec_cfg, args=((cfg, start_timestamp)))
        pool.close()
        pool.join()

    else:
        for cfg in cfgs:
            exec_cfg(cfg, start_timestamp)

def exec_cfg(cfg, start_timestamp):
    if cfg.repeat_experiments > 1:
        base_folder = time.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        base_folder = None
    if cfg.multiple_models != None:
        base_folder = (
            start_timestamp
            + "_"
            + cfg.ctx[: cfg.ctx.find("2021")]
            + cfg.multiple_models[cfg.multiple_models.rfind("/") + 1 :]
        )

    for repetition in range(cfg.repeat_experiments):
        check_cfg_consistency(cfg)
        dataset, algorithm, eval_inst, evals = load_objects_cfgs(
            #cfg, base_folder=base_folder + "_" + str(repetition)
            cfg, base_folder=base_folder)
        if "train" in cfg.mode:
            init_logging(eval_inst.get_run_folder())
            logger = logging.getLogger(__name__)
            algorithm.fit(dataset.train_data(), eval_inst.get_run_folder(), logger)
            algorithm.save(eval_inst.get_run_folder())
            dataset.save(os.path.join(eval_inst.get_run_folder(), "dataset"))

            dataset.save(
                os.path.join(
                    "./models/trained_models/", algorithm.name, "dataset", dataset.name
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
    cfg.to_json(filename=os.path.join(eval_inst.run_folder, 'cfg.json'))
    print(f"Config {cfg.ctx} is done")

def param_cross_val(cfg, num_times=20, test_split=0.3):
    parallel = True
    results = []
    param_lower_bound = cfg.algorithms.validation.range[0]
    param_upper_bound = cfg.algorithms.validation.range[1]
    if parallel:
        pool = mp.Pool(int(mp.cpu_count()/2))
        for param_value in np.linspace(param_lower_bound, param_upper_bound, 20):
            results.append((param_value, pool.apply_async(calc_fixed_param, args=((param_value,
                cfg, num_times, test_split)))))
        pool.close()
        pool.join()
        results_df = pd.DataFrame(map(
            lambda x: (x[0], x[1].get()[0], x[1].get()[1], x[1].get()[2]),
            results),
            columns=['param', 'error_mean', 'gaussian_mean', 'gaussian_var'])
    else:
        for param_value in np.linspace(param_lower_bound, param_upper_bound, 2):
            results.append((param_value, calc_fixed_param(param_value,
                cfg, num_times, test_split)))
        results_df = pd.DataFrame(map(
            lambda x: (x[0], x[1][0], x[1][1], x[1][2]),
            results),
            columns=['param', 'error_mean', 'gaussian_mean', 'gaussian_var'])


    #import pdb; pdb.set_trace()
    # find largest mean with smallest var (not just largest mean)
    max_gaussian_mean = results_df.loc[results_df['gaussian_mean'].idxmax()]
    return (max_gaussian_mean['param'], max_gaussian_mean['error_mean'],
            max_gaussian_mean['gaussian_mean'],
            max_gaussian_mean['gaussian_var'], results_df)



def calc_fixed_param(param_value, cfg, num_times, test_split):
    parameter = cfg.algorithms.validation.parameter
    dataset = load_dataset(cfg)
    X = dataset.train_data()

    algorithm = load_algorithm(cfg)
    algorithm.__dict__[parameter] = param_value
    split = ShuffleSplit(n_splits=num_times, train_size=1 - test_split)
    #self.__dict__[parameter] = param_value
    mean_error_list = []
    gaussian_nb_list = []
    for train_ds, test_ds in split.split(X):
        algorithm.fit(X.loc[train_ds])
        res = algorithm.predict(X.loc[test_ds])
        mean_error = ((X.loc[test_ds] - res) ** 2).sum(axis=1).mean()
        gnb_acc_latent = algorithm.calc_naive_bayes(
            X.loc[train_ds],
            X.loc[test_ds],
            dataset.train_labels[train_ds],
            dataset.train_labels[test_ds],
        )
        gaussian_nb_list.append(gnb_acc_latent)
        mean_error_list.append(mean_error)
        algorithm.module.apply(reset_weights)
    param_error_mean = np.array(mean_error_list).mean()
    gaussian_acc_mean = np.array(gaussian_nb_list).mean()
    gaussian_acc_var = np.var(np.array(gaussian_nb_list))
    print(f"Param_value: {param_value}")
    print(f"Gaussian: {gaussian_acc_mean}")
    print(f"GaussianVar: {gaussian_acc_var}")
    return param_error_mean, gaussian_acc_mean, gaussian_acc_var




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
        model_containing_folder = cfgs[-1].multiple_models
        for _ in range(len(os.listdir(model_containing_folder)) - 1):
            cfgs.append(copy.deepcopy(cfgs[0]))
        for cfg, model_folder in zip(cfgs, os.listdir(model_containing_folder)):
            model_path = model_containing_folder + "/" + model_folder
            cfg.algorithm = model_path
            cfg.ctx = cfg.ctx + "_" + model_path[model_path.rfind("/") + 1 :]
            dataset_path = model_path + "/dataset"
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
                if cfg_dataset[0] == dataset_type:
                    cfg_dataset[1].file_path = dataset_path
    return cfgs


def check_cfg_consistency(cfg):
    pass
    # check data dim = input dim
    # check if for each evaluation linsubfctcount is required
    # if no train, then you must have a load field


def load_objects_cfgs(cfg, base_folder):
    dataset = load_dataset(cfg)
    algorithm = load_algorithm(cfg)
    eval_inst, evals = load_evals(cfg, base_folder)
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
            num_testpoints=cfg.datasets.synthetic_test_samples,
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
    elif cfg.dataset == "electricDevices":
        dataset = electricDevices(
            file_path=cfg.datasets.electricDevices.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "italyPowerDemand":
        dataset = italyPowerDemand(
            file_path=cfg.datasets.italyPowerDemand.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "ecg5000":
        dataset = ecg5000(
            file_path=cfg.datasets.ecg5000.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "chinatown":
        dataset = chinatown(
            file_path=cfg.datasets.chinatown.file_path,
            subsample=cfg.datasets.subsample,
        )
    elif cfg.dataset == "crop":
        dataset = crop(
            file_path=cfg.datasets.crop.file_path,
            subsample=cfg.datasets.subsample,
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
    elif cfg.dataset == "syntheticControl":
        dataset = syntheticControl(
            file_path=cfg.datasets.syntheticControl.file_path,
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
                topology=[2, 1, 2],
                fct_dist=cfg.algorithms.autoencoder.fct_dist,
                border_dist=cfg.algorithms.autoencoder.border_dist,
                num_border_points=cfg.algorithms.autoencoder.num_border_points,
            )
            algorithm.load(cfg.algorithm)
    elif cfg.algorithm == "autoencoder":
        algorithm = autoencoder(
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
            dropout=cfg.algorithms.autoencoder.dropout,
            L2Reg=cfg.algorithms.autoencoder.L2Reg,
            num_border_points=cfg.algorithms.autoencoder.num_border_points,
        )
    else:
        raise Exception("Could not load algorithm")
    return algorithm


def load_evals(cfg, base_folder=None):
    eval_inst = evaluation(base_folder)
    eval_inst.make_run_folder(ctx=cfg.ctx)
    evals = []
    if "parallelQualplots" in cfg.evaluations:
        evals.append(parallelQualplots(eval_inst=eval_inst))
    if "downstream_kmeans" in cfg.evaluations:
        evals.append(downstream_kmeans(eval_inst=eval_inst))
    if "downstream_naiveBayes" in cfg.evaluations:
        evals.append(downstream_naiveBayes(eval_inst=eval_inst))
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
    if "marabou_runtime" in cfg.evaluations:
        evals.append(marabou_runtime(eval_inst=eval_inst))
    if "bias_feature_imp" in cfg.evaluations:
        evals.append(bias_feature_imp(eval_inst=eval_inst))
    if "interpolation_func_diffs_pairs" in cfg.evaluations:
        evals.append(interpolation_func_diffs_pairs(eval_inst=eval_inst))
    if "interpolation_error_plot" in cfg.evaluations:
        evals.append(interpolation_error_plot(eval_inst=eval_inst))
    if "mnist_interpolation_func_diffs_pairs" in cfg.evaluations:
        evals.append(mnist_interpolation_func_diffs_pairs(eval_inst=eval_inst))
    if "tsne_latent" in cfg.evaluations:
        evals.append(tsne_latent(eval_inst=eval_inst))
    if "orig_latent_dist_ratios" in cfg.evaluations:
        evals.append(orig_latent_dist_ratios(eval_inst=eval_inst))
    return eval_inst, evals



if __name__ == "__main__":
    main()
