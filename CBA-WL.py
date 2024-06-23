import os
import json

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import KFold, train_test_split
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


class RuleGenerator:
    def __init__(self, min_support=0.1, min_confidence=0.5, min_lift=1):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift

    def getFrequentItemsets(self, data):
        frequent_itemsets = apriori(data, min_support=self.min_support, use_colnames=True)

        def formatFrequentItemsets(frequent_itemsets):
            formatted_itemsets = []
            for index, row in frequent_itemsets.iterrows():
                itemset = []
                for i, item in enumerate(row['itemsets']):
                    key = item.split(': ')[0]
                    value = item.split(': ')[1]
                    itemset.append({key: value})
                formatted_itemsets.append(itemset)
            return formatted_itemsets

        frequent_itemsets = formatFrequentItemsets(frequent_itemsets)

        return frequent_itemsets

    def generateRule(self, data, frequent_itemset):
        target_name = data.columns[-1]
        targets = data[target_name].unique()
        rule = None
        max_harmonic_mean = 0
        for target in targets:
            candidate_rule = {'features': frequent_itemset, 'target': target}
            support, confidence, lift = self.computeSupportAndConfidenceAndLift(
                data, candidate_rule)
            if (confidence >= self.min_confidence and lift >= self.min_lift):
                # harmonic_mean = 2 / (1/confidence + 1/lift)
                harmonic_mean = confidence
                if harmonic_mean > max_harmonic_mean:
                    max_harmonic_mean = harmonic_mean
                    rule = candidate_rule
                    rule['support'] = support
                    rule['confidence'] = confidence
                    rule['lift'] = lift
                    rule['hm'] = harmonic_mean
        return rule

    def computeSupportAndConfidenceAndLift(self, data, rule):
        target_name = data.columns[-1]

        features = rule['features']
        target = rule['target']

        condition = data[target_name] == target
        condition_without_target = None
        for item in features:
            key = list(item.keys())[0]
            value = list(item.values())[0]
            condition = condition & (data[key] == value)
            condition_without_target = condition_without_target & (
                data[key] == value) if condition_without_target is not None else (data[key] == value)
        if data[condition_without_target].shape[0] == 0:
            return 0, 0, 0
        support = data[condition].shape[0] / data.shape[0]
        confidence = data[condition].shape[0] / data[condition_without_target].shape[0]
        lift = confidence / (data[data[target_name] == target].shape[0] / data.shape[0])
        return support, confidence, lift


class CBAWL:
    def __init__(self, dataset_id, min_support=0.1, min_confidence=0.5, min_lift=1):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.dataset_id = dataset_id
        self.dataHandler = DataHandler(dataset_id)
        self.ruleGenerator = RuleGenerator(min_support, min_confidence, min_lift)

    def deatasetCover(self, rules, data):
        target_name = data.columns[-1]

        def cover(rule, data, target_name):
            features = rule['features']
            target = rule['target']
            condition = data[target_name] == target
            for item in features:
                key = list(item.keys())[0]
                value = list(item.values())[0]
                condition = condition & (data[key] == value)
            if data[condition].shape[0] == 0:
                return None
            return data[condition]
        data_ = data
        strong_rules = []
        weak_rules = []
        for rule in rules:
            if data_.shape[0] == 0:
                weak_rules.append(rule)
                continue
            cover_data = cover(rule, data_, target_name)
            if cover_data is not None:
                data_ = data_.drop(cover_data.index)
                strong_rules.append(rule)
            else:
                weak_rules.append(rule)

        default_class = None

        if data_.shape[0] != 0:
            default_class = data_[target_name].value_counts().idxmax()
        else:
            # strong_rules裡面最多的target
            if len(strong_rules) != 0:
                default_class = max(strong_rules, key=lambda x: x['target'])
            else:
                default_class = max(weak_rules, key=lambda x: x['target'])

        if default_class is None:
            default_class = data[target_name].value_counts().idxmax()

        return strong_rules, weak_rules, default_class

    def model(self, frequent_itemsets, data):
        rules = []
        for frequent_itemset in frequent_itemsets:
            rule = self.ruleGenerator.generateRule(data, frequent_itemset)
            if rule is not None:
                rules.append(rule)

        strong_rules, weak_rules, default_class = self.deatasetCover(rules, data)

        return strong_rules, weak_rules, default_class

    def predict(self, strong_rules, weak_rules, default_class, data):
        def checkInstance(rule, instance):
            features = rule['features']
            for item in features:
                key = list(item.keys())[0]
                value = list(item.values())[0]
                if instance[key] != value:
                    return False
            return True

        prediction = []

        for index, row in data.iterrows():
            for rule in strong_rules:
                if checkInstance(rule, row):
                    prediction.append(rule['target'])
                    break
            else:
                for rule in weak_rules:
                    if checkInstance(rule, row):
                        prediction.append(rule['target'])
                        break
                else:
                    prediction.append(default_class)
        return prediction


if __name__ == '__main__':
    dataset_id = 53

    cbawl = CBAWL(dataset_id, min_support=0.1, min_confidence=0.5, min_lift=1)
    data = cbawl.dataHandler.loadData()
    features = cbawl.dataHandler.oneHotEncoding(data)
    features_importance = cbawl.dataHandler.getFeaturesImportance(data)
    features = cbawl.dataHandler.delLowImportanceFeatures(data, features_importance)
    X = features
    y = data.iloc[:, -1]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train_te = cbawl.dataHandler.oneHotEncoding(X_train)
        frequent_itemsets = cbawl.ruleGenerator.getFrequentItemsets(X_train_te)
        strong_rules, weak_rules, default_class = cbawl.model(frequent_itemsets, X_train)
        y_pred = cbawl.predict(strong_rules, weak_rules, default_class, X_test)

        print(classification_report(y_test, y_pred, zero_division=0))
