from mlxtend.frequent_patterns import apriori
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from DataHandler import DataHandler


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
                harmonic_mean = 2 / (1/confidence + 1/lift)
                # harmonic_mean = confidence
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

        condSupportCount = data[condition_without_target].shape[0]
        ruleSupportCount = data[condition].shape[0]
        targetCount = data[data[target_name] == target].shape[0]
        dataCount = data.shape[0]

        if condSupportCount == 0 or ruleSupportCount == 0 or targetCount == 0 or dataCount == 0:
            return 0, 0, 0

        support = ruleSupportCount / dataCount
        confidence = ruleSupportCount / condSupportCount
        lift = confidence / (targetCount / dataCount)
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
    dataset_id = 19

    cbawl = CBAWL(dataset_id, min_support=0.01, min_confidence=0.5, min_lift=1)
    data = cbawl.dataHandler.loadData()
    features_importance = cbawl.dataHandler.getFeaturesImportance(data)
    features = data.drop(data.columns[-1], axis=1)
    features = cbawl.dataHandler.delLowImportanceFeatures(features, features_importance)
    X_train_te = cbawl.dataHandler.oneHotEncoding(features)
    frequent_itemsets = cbawl.ruleGenerator.getFrequentItemsets(X_train_te)
    strong_rules, weak_rules, default_class = cbawl.model(frequent_itemsets, data)

    for rule in strong_rules:
        print(rule)

    print("Default class: ", default_class)
    # y_pred = cbawl.predict(strong_rules, weak_rules, default_class, X_test)

    # print(classification_report(y_test, y_pred, zero_division=0))
