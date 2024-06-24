import json
import os

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo


class DataHandler:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id

        self.file_dir_path = "./data"
        if not os.path.exists(self.file_dir_path):
            os.makedirs(self.file_dir_path)

        f = open('dataset_list.json', 'r')
        dataset_list = f.read()
        dataset_list = json.loads(dataset_list)

        self.dataset_list = {int(key): value for key, value in dataset_list.items()}

    def loadData(self):
        try:
            data_name = self.dataset_list[self.dataset_id]
        except KeyError:
            date = fetch_ucirepo(id=self.dataset_id)
            if date is None:
                print("Dataset not found")
                exit()
            else:
                data_name = (date.metadata['name'])
                self.dataset_list[self.dataset_id] = data_name
                with open('dataset_list.json', 'w') as f:
                    json.dump(self.dataset_list, f)

        data_file_path = f"{self.file_dir_path}/{data_name}.csv"
        if not os.path.exists(data_file_path):
            url = f"https://archive.ics.uci.edu/static/public/{self.dataset_id}/data.csv"
            data = pd.read_csv(url)
            data.to_csv(data_file_path, index=False)
        else:
            data = pd.read_csv(data_file_path)

        return data

    def oneHotEncoding(self, data):

        def formatData(data):
            return [[f"{col}: {value}" for col, value in row.items()] for _, row in data.iterrows()]

        data_list = formatData(data)
        te = TransactionEncoder()
        te_ary = te.fit(data_list).transform(data_list)
        features_te = pd.DataFrame(te_ary, columns=te.columns_)

        return features_te

    def getFeaturesImportance(self, data):

        def mappingFeaturesWithImportance(features_name, feature_importances):
            features_importance = {}
            for index, feature_name in enumerate(features_name):
                features_importance[feature_name] = float(feature_importances[index])
            return features_importance

        features_name = data.columns[:-1]
        label_encoder = LabelEncoder()
        encode_data = data.apply(label_encoder.fit_transform)
        X = encode_data.iloc[:, :-1]
        y = encode_data.iloc[:, -1]
        model = RandomForestClassifier()
        model.fit(X, y)
        feature_importances = model.feature_importances_
        feature_importances = feature_importances.round(3)
        feature_importances = mappingFeaturesWithImportance(features_name, feature_importances)

        return feature_importances

    def delLowImportanceFeatures(self, data, features_importance):
        def getThresholdByMean(features_importance):
            mean = np.mean(features_importance)
            threshold = mean
            return threshold

        def getThresholdByMedian(features_importance):
            median = np.median(features_importance)
            threshold = median-0.1

            return threshold

        features_importance_list = list(features_importance.values())
        threshold = getThresholdByMedian(features_importance_list)

        for feature, importance in features_importance.items():
            if importance < threshold:
                data = data.drop(columns=[feature])
        return data
