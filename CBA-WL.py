import os

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

DATASETS = {
    19: 'Car Evaluation',
}


class DataPreprocessing:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.data, self.features, self.target = self.loadData()

    def loadData(self):

        file_dir_path = "./data"

        if not os.path.exists(file_dir_path):
            os.makedirs(file_dir_path)

        data_name = DATASETS[self.dataset_id]
        if not os.path.exists(f"{file_dir_path}/{data_name}.csv"):
            url = f"https://archive.ics.uci.edu/static/public/{self.dataset_id}/data.csv"
            data = pd.read_csv(url)
            data.to_csv(f"{file_dir_path}/{data_name}.csv", index=False)
        else:
            data = pd.read_csv(f"{file_dir_path}/{data_name}.csv")

        # features = dataset.data.features
        # target = dataset.data.targets
        # return dataset.data, features, target


if __name__ == '__main__':
    dataset_id = 19
    data = DataPreprocessing(dataset_id)
