from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np


class AdultIncomeData:
    def __init__(self, seed):
        # column names for the dataset
        self.column_names = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        self.target = "income"  # Last attribute
        self.groups = [
            "age",
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
        ]  # First 8 attributes

        # loading the dataset from the UCI repository
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        )
        self.df = pd.read_csv(url, header=None, names=self.column_names)

        # handling missing values
        self.df.replace(" ?", pd.NA, inplace=True)  # replace ' ?' with NA
        self.df.dropna(inplace=True)  # drop NA values

        # separate the predictors (features) and the target (label)
        self.features = self.df.drop(self.target, axis=1)
        self.target = self.df[self.target]

        # one-hot encoding of the labels
        encoder = OneHotEncoder(sparse=False)
        self.target_1hot = encoder.fit_transform(self.target.values.reshape(-1, 1))

        # split the data into training set and test set
        (
            self.features_train,
            self.features_test,
            self.target_train,
            self.target_test,
        ) = train_test_split(
            self.features, self.target_1hot, test_size=0.2, random_state=seed
        )
        self.features_train = self.features_train.reset_index(drop=True)
        self.features_test = self.features_test.reset_index(drop=True)

        self.group_keys = []
        for g in self.groups:
            self.group_keys += self.features[g].unique().tolist()

    def get_groups(self, feature, data):
        return {
            value: list(data[data[feature] == value].index) for value in self.group_keys
        }

    def get_training_data(self):
        self.groups_train = {}
        for g in self.groups:
            self.groups_train = {
                **self.groups_train,
                **self.get_groups(g, self.features_train),
            }

        groups_train = [self.groups_train[key] for key in self.group_keys]
        return (
            np.array(self.features_train),
            np.array(self.target_train),
            np.array(groups_train),
        )

    def get_test_data(self):
        self.groups_test = {}
        for g in self.groups:
            self.groups_test = {
                **self.groups_test,
                **self.get_groups(g, self.features_test),
            }

        groups_test = [self.groups_test[key] for key in self.group_keys]
        return (
            np.array(self.features_test),
            np.array(self.target_test),
            np.array(groups_test),
        )

class BankMarketingData:
    def __init__(self, seed):
        self.target = "y"  # Last attribute
        self.groups = [
            "age",
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
        ]  # First 8 attributes

        # loading the dataset from the UCI repository
        self.df = pd.read_csv('./bank-full.csv', sep=";")

        # handling missing values
        self.df.replace(" ?", pd.NA, inplace=True)  # replace ' ?' with NA
        self.df.dropna(inplace=True)  # drop NA values
        self.df['age'] = self.df['age'].astype(int).apply(lambda x: round(x / 5) * 5)

        # separate the predictors (features) and the target (label)
        self.features = self.df.drop(self.target, axis=1)
        self.target = self.df[self.target]

        # one-hot encoding of the labels
        encoder = OneHotEncoder(sparse=False)
        self.target_1hot = encoder.fit_transform(self.target.values.reshape(-1, 1))

        # split the data into training set and test set
        (
            self.features_train,
            self.features_test,
            self.target_train,
            self.target_test,
        ) = train_test_split(
            self.features, self.target_1hot, test_size=0.2, random_state=seed
        )
        self.features_train = self.features_train.reset_index(drop=True)
        self.features_test = self.features_test.reset_index(drop=True)

        self.group_keys = []
        for g in self.groups:
            self.group_keys += self.features[g].unique().tolist()

    def get_groups(self, feature, data):
        return {
            value: list(data[data[feature] == value].index) for value in self.group_keys
        }

    def get_training_data(self):
        self.groups_train = {}
        for g in self.groups:
            self.groups_train = {
                **self.groups_train,
                **self.get_groups(g, self.features_train),
            }

        groups_train = [self.groups_train[key] for key in self.group_keys]
        return (
            np.array(self.features_train),
            np.array(self.target_train),
            np.array(groups_train),
        )

    def get_test_data(self):
        self.groups_test = {}
        for g in self.groups:
            self.groups_test = {
                **self.groups_test,
                **self.get_groups(g, self.features_test),
            }

        groups_test = [self.groups_test[key] for key in self.group_keys]
        return (
            np.array(self.features_test),
            np.array(self.target_test),
            np.array(groups_test),
        )


class DryBeanData:
    def __init__(self, seed):
        self.target = "Class"  # Last attribute
        self.groups = [   # The first 8 attributes for group_keys might be changed based on your specific use case
            "Area",
            "Perimeter",
            "MajorAxisLength",
            "MinorAxisLength",
            "AspectRation",
            "Eccentricity",
            "ConvexArea",
            "EquivDiameter",
        ]  

        # loading the dataset from the UCI repository
        # replace with the correct url or local file path to your data
        self.df = pd.read_csv('./Dry_Bean_Dataset.csv')

        # handling missing values
        self.df.replace(" ?", pd.NA, inplace=True)  # replace ' ?' with NA
        self.df.dropna(inplace=True)  # drop NA values

        # separate the predictors (features) and the target (label)
        self.features = self.df.drop(self.target, axis=1)
        self.target = self.df[self.target]

        for group in self.groups:
            self.features[group] = pd.cut(self.features[group], 10)

        # one-hot encoding of the labels
        encoder = OneHotEncoder(sparse=False)
        self.target_1hot = encoder.fit_transform(self.target.values.reshape(-1, 1))

        # split the data into training set and test set
        (
            self.features_train,
            self.features_test,
            self.target_train,
            self.target_test,
        ) = train_test_split(
            self.features, self.target_1hot, test_size=0.2, random_state=seed
        )
        self.features_train = self.features_train.reset_index(drop=True)
        self.features_test = self.features_test.reset_index(drop=True)

        self.group_keys = []
        for g in self.groups:
            self.group_keys += self.features[g].unique().tolist()

    def get_groups(self, feature, data):
        return {
            value: list(data[data[feature] == value].index) for value in self.group_keys
        }

    def get_training_data(self):
        self.groups_train = {}
        for g in self.groups:
            self.groups_train = {
                **self.groups_train,
                **self.get_groups(g, self.features_train),
            }

        groups_train = [self.groups_train[key] for key in self.group_keys]
        return (
            np.array(self.features_train),
            np.array(self.target_train),
            np.array(groups_train),
        )

    def get_test_data(self):
        self.groups_test = {}
        for g in self.groups:
            self.groups_test = {
                **self.groups_test,
                **self.get_groups(g, self.features_test),
            }

        groups_test = [self.groups_test[key] for key in self.group_keys]
        return (
            np.array(self.features_test),
            np.array(self.target_test),
            np.array(groups_test),
        )


if __name__ == "__main__":
    # Example of how to use the class
    data = BankMarketingData(10)
    features_train, target_train, groups_train = data.get_training_data()
    features_test, target_test, groups_test = data.get_test_data()

    print("Training set size:", len(features_train))
    print("Test set size:", len(features_test))
    print("Number of training groups:", len(groups_train))
    print("Number of test groups:", len(groups_test))
    print("Class size", len(target_train[0]))

    data = AdultIncomeData(10)
    features_train, target_train, groups_train = data.get_training_data()
    features_test, target_test, groups_test = data.get_test_data()

    print("Training set size:", len(features_train))
    print("Test set size:", len(features_test))
    print("Number of training groups:", len(groups_train))
    print("Number of test groups:", len(groups_test))
    print("Class size", len(target_train[0]))

    data = DryBeanData(10)
    features_train, target_train, groups_train = data.get_training_data()
    features_test, target_test, groups_test = data.get_test_data()

    print("Training set size:", len(features_train))
    print("Test set size:", len(features_test))
    print("Number of training groups:", len(groups_train))
    print("Number of test groups:", len(groups_test))
    print("Class size", len(target_train[0]))