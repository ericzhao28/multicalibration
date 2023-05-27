import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import rc

# Use LaTeX for text rendering
rc("text", usetex=True)

# Set Seaborn aesthetic parameters to defaults
sns.set()

# Read the data
df = pd.read_csv("new_bank_market_results.csv")

# Assuming df is your DataFrame
# Group by Algorithm, Learning Rate, Dataset, and Iterations
grouped_train = df.groupby(["Algorithm", "Learning Rate", "Dataset", "Iterations"])
grouped_test = df.groupby(["Algorithm", "Learning Rate", "Dataset", "Iterations"])
grouped_avg_test = df.groupby(["Algorithm", "Learning Rate", "Dataset", "Iterations"])

print(df.isnull().values.any())

# Compute mean and standard deviation of Training Calibration Error
summary_train = grouped_train["Training Calibration Error"].agg(["mean", "sem", "count"])

# Compute mean and standard deviation of Testing Calibration Error
summary_test = grouped_test["Testing Calibration Error"].agg(["mean", "sem", "count"])

# Compute mean and standard deviation of Testing Calibration Error
summary_avg_test = grouped_avg_test["Testing Calibration Error (Ergodic)"].agg(["mean", "sem", "count"])


# Reset the index for the next steps
summary_train.reset_index(inplace=True)
summary_test.reset_index(inplace=True)
summary_avg_test.reset_index(inplace=True)

# Initialize an empty DataFrame to hold the final results
final_summary = pd.DataFrame()

# Loop over each Algorithm and Dataset
for algorithm in df["Algorithm"].unique():
    for dataset in df["Dataset"].unique():
        # Subset the data for this Algorithm and Dataset
        subset_train = summary_train[
            (summary_train["Algorithm"] == algorithm)
            & (summary_train["Dataset"] == dataset)
        ]
        subset_test = summary_test[
            (summary_test["Algorithm"] == algorithm)
            & (summary_test["Dataset"] == dataset)
        ]
        subset_avg_test = summary_avg_test[
            (summary_avg_test["Algorithm"] == algorithm)
            & (summary_avg_test["Dataset"] == dataset)
        ]
        if algorithm == "Prod-Prod":
            print(subset_train.isnull().any())
            print(subset_train[subset_train.isnull().any(axis=1)])

        # Find the last iteration for each Learning Rate
        last_iteration_train = subset_train.groupby(["Learning Rate"])[
            "Iterations"
        ].max()
        last_iteration_test = subset_test.groupby(["Learning Rate"])[
            "Iterations"
        ].max()
        last_iteration_avg_test = subset_avg_test.groupby(["Learning Rate"])[
            "Iterations"
        ].max()
        last_iteration_summary_train = subset_train[
            subset_train["Iterations"].isin(last_iteration_train)
        ]
        last_iteration_summary_test = subset_test[
            subset_test["Iterations"].isin(last_iteration_test)
        ]
        last_iteration_summary_avg_test = subset_train[
            subset_avg_test["Iterations"].isin(last_iteration_avg_test)
        ]

        # Find the Learning Rate that yields the minimum mean Training Calibration Error
        best_lr_summary_train = last_iteration_summary_train

        # Find the corresponding Testing Calibration Error
        best_lr_summary_test = last_iteration_summary_test

        # Find the corresponding Avg Testing Calibration Error
        best_lr_summary_avg_test = last_iteration_summary_avg_test

        # Merge the Training and Testing summaries
        best_lr_summary = pd.merge(
            pd.merge(
                best_lr_summary_train,
                best_lr_summary_test,
                on=["Algorithm", "Learning Rate", "Dataset", "Iterations"],
                suffixes=("_train", "_test"),
            ),
            best_lr_summary_avg_test,
            on=["Algorithm", "Learning Rate", "Dataset", "Iterations"],
            suffixes=("", "_avg_test"),
        )
        best_lr_summary = best_lr_summary.rename(
            columns={"mean": "mean_avg_test", "sem": "sem_avg_test",  "count": "count_avg_test"}
        )

        # Append this to the final summary
        final_summary = final_summary.append(best_lr_summary, ignore_index=True)


# Convert the mean and standard deviation columns to the "... ± ..." format
# Add count information to the final summary
final_summary["Training Calibration Error"] = final_summary.apply(
    lambda row: f"{row['mean_train']:.3f} $\pm$ {row['sem_train']:.3f} (n={row['count_train']})", axis=1
)
final_summary["Testing Calibration Error"] = final_summary.apply(
    lambda row: f"{row['mean_test']:.3f} $\pm$ {row['sem_test']:.3f} (n={row['count_test']})", axis=1
)
final_summary["Testing Calibration Error (Ergodic)"] = final_summary.apply(
    lambda row: f"{row['mean_avg_test']:.6f} $\pm$ {row['sem_avg_test']:.6f} (n={row['count_avg_test']})", axis=1
)

# Drop the original mean, sem and count columns
final_summary.drop(
    columns=["mean_train", "sem_train", "count_train", 
             "mean_test", "sem_test", "count_test", 
             "mean_avg_test", "sem_avg_test", "count_avg_test"], inplace=True
)

# Reorder the columns
final_summary = final_summary[
    [
        "Algorithm",
        "Dataset",
        "Learning Rate",
        "Training Calibration Error",
        "Testing Calibration Error",
        "Testing Calibration Error (Ergodic)",
    ]
]

# Convert the DataFrame to
latex_table = final_summary.to_latex(index=False, escape=False)
print(latex_table)

