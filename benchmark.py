import numpy as np
import random
import csv
from hedge import Hedge, MLProd, OnlineGradientDescent, OptimisticHedge
from multicalibration import update_hedge_algorithms, Adversary
from dataset import AdultIncomeData, BankMarketingData, DryBeanData
from calibration import (
    discretize_values,
    calculate_average_labels,
    calculate_calibration_errors,
)
import pickle


FREQUENCY = 12 #3


def run_expt(
    dataset,
    seed,
    other_alg_class,
    other_name,
    alg_class,
    name,
    other_lr,
    lr,
    n_bins,
    features_train,
    target_train,
    groups_train,
    features_test,
    target_test,
    groups_test,
    writer,
    iterations=50,
    base_lr=1,
    base_other_lr=100,
):
    lr_full = str(lr) + "+" + str(other_lr)
    title = (
        lambda x: str(seed)
        + "_"
        + dataset
        + "_"
        + name
        + "_"
        + other_name
        + "_"
        + str(lr)
        + "_"
        + str(other_lr)
        + "_"
        + "1"
        + "_"
        + str(x)
    )

    historical_test_error = []

    i = 1
    adv = None
    while True:
        if i % FREQUENCY == 1:
            try:
                train_algs = pickle.load(
                    open("multicalibrate_train_" + title(i) + ".pkl", "rb")
                )
                test_algs = pickle.load(
                    open("multicalibrate_test_" + title(i) + ".pkl", "rb")
                )
                if other_name:
                    adv = pickle.load(
                        open("multicalibrate_adversary" + title(i) + ".pkl", "rb")
                    )
            except Exception as e:
                print(e)
                break

            predictions = np.array([alg.weights for alg in train_algs])
            bins = discretize_values(predictions, n_bins)
            avg_train_y, counts = calculate_average_labels(
                target_train, bins, groups_train, n_bins
            )

            errors = calculate_calibration_errors(
                avg_train_y, predictions, bins, groups_train
            )
            weighted_errors = (
                errors * counts[:, :, np.newaxis, np.newaxis] / len(target_train)
            )
            train_error = weighted_errors.max()
            # train_error = np.quantile(weighted_errors, 0.99999)

            predictions = np.array([alg.weights for alg in test_algs])
            bins = discretize_values(predictions, n_bins)
            avg_test_y, counts = calculate_average_labels(
                target_test, bins, groups_test, n_bins
            )

            errors = calculate_calibration_errors(
                avg_test_y, predictions, bins, groups_test
            )
            weighted_errors = (
                errors * counts[:, :, np.newaxis, np.newaxis] / len(target_test)
            )
            historical_test_error.append(weighted_errors)
            test_error = weighted_errors.max()
            # test_error = np.quantile(weighted_errors, 0.99999)

            avg_test_error = np.mean(weighted_errors, axis=0).max()

            print(
                "(" + title(i) + ")",
                "Train: ",
                train_error,
                "Test: ",
                test_error,
                "Average Test Error:",
                avg_test_error,
            )

            writer.writerow(
                {
                    "Algorithm": name,
                    "Dataset": dataset,
                    "Iterations": i,
                    "Learning Rate": lr_full,
                    "Training Calibration Error": train_error,
                    "Testing Calibration Error": test_error,
                    "Testing Calibration Error (Ergodic)": avg_test_error,
                }
            )
            csvfile.flush()
        i += 1

    learning_rate = lambda x: base_lr * np.power(lr, x)
    other_learning_rate = lambda x: base_other_lr * np.power(other_lr, x)

    train_algs = [
        alg_class(len(target_train[0]), learning_rate=learning_rate(i))
        for _ in features_train
    ]
    test_algs = [
        alg_class(len(target_test[0]), learning_rate=learning_rate(i))
        for _ in features_test
    ]
    while i <= iterations:
        predictions = np.array([alg.weights for alg in train_algs])
        bins = discretize_values(predictions, n_bins)
        avg_train_y, counts = calculate_average_labels(
            target_train, bins, groups_train, n_bins
        )
        if adv is None:
            adv = Adversary(
                other_alg_class,
                avg_train_y.flatten().shape[0] * 2,
                other_learning_rate(i),
            )

        (
            bad_bin,
            bad_group,
            bad_class,
            underestimate,
        ), train_error = adv.find_bad_bin_group(
            avg_train_y, predictions, counts, bins, groups_train, other_learning_rate(i)
        )

        _ = update_hedge_algorithms(
            train_algs,
            bad_bin,
            bad_group,
            bad_class,
            underestimate,
            bins,
            groups_train,
            learning_rate(i),
        )
        _ = update_hedge_algorithms(
            test_algs,
            bad_bin,
            bad_group,
            bad_class,
            underestimate,
            bins,
            groups_test,
            learning_rate(i),
        )

        if i % FREQUENCY == 1:
            predictions = np.array([alg.weights for alg in test_algs])
            bins = discretize_values(predictions, n_bins)
            avg_test_y, counts = calculate_average_labels(
                target_test, bins, groups_test, n_bins
            )

            errors = calculate_calibration_errors(
                avg_test_y, predictions, bins, groups_test
            )
            weighted_errors = (
                errors * counts[:, :, np.newaxis, np.newaxis] / len(target_test)
            )
            historical_test_error.append(weighted_errors)
            test_error = weighted_errors.max()
            avg_test_error = np.mean(weighted_errors, axis=0).max()

            print(
                "(" + title(i) + ")",
                "Train: ",
                train_error,
                "Test: ",
                test_error,
                "Average Test Error:",
                avg_test_error,
            )

            writer.writerow(
                {
                    "Algorithm": name,
                    "Dataset": dataset,
                    "Iterations": i,
                    "Learning Rate": lr_full,
                    "Training Calibration Error": train_error,
                    "Testing Calibration Error": test_error,
                    "Testing Calibration Error (Ergodic)": avg_test_error,
                }
            )
            pickle.dump(
                train_algs, open("multicalibrate_train_" + title(i) + ".pkl", "wb")
            )
            pickle.dump(
                test_algs, open("multicalibrate_test_" + title(i) + ".pkl", "wb")
            )

            if other_name:
                pickle.dump(
                    adv, open("multicalibrate_adversary" + title(i) + ".pkl", "wb")
                )

            csvfile.flush()
        else:
            print("(" + title(i) + ")", "Train: ", train_error)

        i += 1


# Get command line argument if exists and None if not
import sys

if len(sys.argv) > 1:
    it_todo = sys.argv[1]
else:
    it_todo = None

if True:
    with open("new_sweep_adult_income_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Algorithm",
            "Iterations",
            "Dataset",
            "Learning Rate",
            "Training Calibration Error",
            "Testing Calibration Error",
            "Testing Calibration Error (Ergodic)",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for seed in range(10):
            dataset = "AdultIncome"
            np.random.seed(seed)
            random.seed(seed)
            data = AdultIncomeData(seed)
            features_train, target_train, groups_train = data.get_training_data()
            features_test, target_test, groups_test = data.get_test_data()
            n_bins = 10

            for other_lr in [0.98, 0.99, 0.95, 0.9]:
                for alg_class, name, lr in [
                    (Hedge, "Hedge-Hedge", 0.95),
                    (Hedge, "Hedge-Hedge", 0.9),
                    (OptimisticHedge, "OGD-OGD", 0.95),
                    (OptimisticHedge, "OGD-OGD", 0.9),
                ]:
                    run_expt(
                        dataset,
                        seed,
                        alg_class,
                        name,
                        alg_class,
                        name,
                        other_lr,
                        lr,
                        n_bins,
                        features_train,
                        target_train,
                        groups_train,
                        features_test,
                        target_test,
                        groups_test,
                        writer,
                    )

            for lr in [0.8, 0.85, 0.9, 0.95]:
                other_alg_class = None
                other_name = ""
                for alg_class, name in [
                    (OptimisticHedge, "OGD"),
                    (Hedge, "Hedge"),
                    (MLProd, "Prod"),
                    (OnlineGradientDescent, "GD"),
                ]:
                    run_expt(
                        dataset,
                        seed,
                        None,
                        "",
                        alg_class,
                        name,
                        0,
                        lr,
                        n_bins,
                        features_train,
                        target_train,
                        groups_train,
                        features_test,
                        target_test,
                        groups_test,
                        writer,
                    )


if True:
    with open("new_adult_income_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Algorithm",
            "Iterations",
            "Dataset",
            "Learning Rate",
            "Training Calibration Error",
            "Testing Calibration Error",
            "Testing Calibration Error (Ergodic)",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for seed in range(20):
            for dataset in ["AdultIncome"]:
                if dataset == "AdultIncome":
                    np.random.seed(seed)
                    random.seed(seed)
                    data = AdultIncomeData(seed)
                    features_train, target_train, groups_train = data.get_training_data()
                    features_test, target_test, groups_test = data.get_test_data()
                    n_bins = 10

                    for alg_class, name, lr, other_lr in [
                        (Hedge, "Hedge-Hedge", 0.95, 0.9),
                        (OptimisticHedge, "OGD-OGD", 0.95, 0.9),
                    ]:
                        run_expt(
                            dataset,
                            seed,
                            alg_class,
                            name,
                            alg_class,
                            name,
                            other_lr,
                            lr,
                            n_bins,
                            features_train,
                            target_train,
                            groups_train,
                            features_test,
                            target_test,
                            groups_test,
                            writer,
                        )

                    other_alg_class = None
                    other_name = ""
                    for alg_class, name, lr in [
                        (OptimisticHedge, "OGD", 0.9),
                        (Hedge, "Hedge", 0.9),
                        (MLProd, "Prod", 0.9),
                        (OnlineGradientDescent, "GD", 0.9),
                    ]:
                        run_expt(
                            dataset,
                            seed,
                            None,
                            "",
                            alg_class,
                            name,
                            0,
                            lr,
                            n_bins,
                            features_train,
                            target_train,
                            groups_train,
                            features_test,
                            target_test,
                            groups_test,
                            writer,
                        )

if True:
    # Dry Bean dataset
    with open("new_dry_beans_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Algorithm",
            "Iterations",
            "Dataset",
            "Learning Rate",
            "Training Calibration Error",
            "Testing Calibration Error",
            "Testing Calibration Error (Ergodic)",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for seed in range(5):
            dataset = "Dry Beans"
            np.random.seed(seed)
            random.seed(seed)
            data = DryBeanData(seed)
            features_train, target_train, groups_train = data.get_training_data()
            features_test, target_test, groups_test = data.get_test_data()
            n_bins = 3
            for lr in [0.8, 0.85, 0.9, 0.95]:
                other_alg_class = None
                other_name = ""
                for alg_class, name in [
                    (MLProd, "Prod"),
                    (OptimisticHedge, "OGD"),
                    (Hedge, "Hedge"),
                    (OnlineGradientDescent, "GD"),
                ]:
                    run_expt(
                        dataset,
                        seed,
                        None,
                        "",
                        alg_class,
                        name,
                        0,
                        lr,
                        n_bins,
                        features_train,
                        target_train,
                        groups_train,
                        features_test,
                        target_test,
                        groups_test,
                        writer,
                        100,
                        2,
                        200,
                    )
            for other_lr in [0.98, 0.99, 0.95, 0.9]:
                for alg_class, name, lr in [
                    (Hedge, "Hedge-Hedge", 0.95),
                    (Hedge, "Hedge-Hedge", 0.99),
                    (Hedge, "Hedge-Hedge", 0.9),
                    (OptimisticHedge, "OGD-OGD", 0.95),
                    (OptimisticHedge, "OGD-OGD", 0.99),
                    (OptimisticHedge, "OGD-OGD", 0.9),
                ]:
                    run_expt(
                        dataset,
                        seed,
                        alg_class,
                        name,
                        alg_class,
                        name,
                        other_lr,
                        lr,
                        n_bins,
                        features_train,
                        target_train,
                        groups_train,
                        features_test,
                        target_test,
                        groups_test,
                        writer,
                        100,
                        2,
                        200,
                    )

if True:
    # Bank Market dataset
    with open("new_bank_market_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "Algorithm",
            "Iterations",
            "Dataset",
            "Learning Rate",
            "Training Calibration Error",
            "Testing Calibration Error",
            "Testing Calibration Error (Ergodic)",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for seed in range(5):
            dataset = "Bank Market"
            np.random.seed(seed)
            random.seed(seed)
            data = BankMarketingData(seed)
            features_train, target_train, groups_train = data.get_training_data()
            features_test, target_test, groups_test = data.get_test_data()
            n_bins = 10

            for other_lr in [0.98, 0.99, 0.95, 0.9]:
                for alg_class, name, lr in [
                    (Hedge, "Hedge-Hedge", 0.95),
                    (Hedge, "Hedge-Hedge", 0.9),
                    (OptimisticHedge, "OGD-OGD", 0.95),
                    (OptimisticHedge, "OGD-OGD", 0.9),
                ]:
                    run_expt(
                        dataset,
                        seed,
                        alg_class,
                        name,
                        alg_class,
                        name,
                        other_lr,
                        lr,
                        n_bins,
                        features_train,
                        target_train,
                        groups_train,
                        features_test,
                        target_test,
                        groups_test,
                        writer,
                    )

            for lr in [0.8, 0.85, 0.9, 0.95]:
                other_alg_class = None
                other_name = ""
                for alg_class, name in [
                    (OptimisticHedge, "OGD"),
                    (Hedge, "Hedge"),
                    (MLProd, "Prod"),
                    (OnlineGradientDescent, "GD"),
                ]:
                    run_expt(
                        dataset,
                        seed,
                        None,
                        "",
                        alg_class,
                        name,
                        0,
                        lr,
                        n_bins,
                        features_train,
                        target_train,
                        groups_train,
                        features_test,
                        target_test,
                        groups_test,
                        writer,
                    )
