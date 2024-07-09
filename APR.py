class APR:
    def __init__(self, data, min_sup, min_conf):
        self.data = data
        self.min_sup = min_sup
        self.min_conf = min_conf

    def genFrequentOne(self):
        data = self.data.copy()
        min_sup = self.min_sup
        min_conf = self.min_conf

        frequent_itemsets = []
        features = data.columns[:-1]
        class_values = data['class'].unique()
        for feature in features:
            for class_value in class_values:
                feature_values = data[feature].unique()
                for value in feature_values:
                    count = data[(data[feature] == value) & (data['class'] == class_value)].shape[0]
                    if count == 0:
                        continue
                    support = count / data.shape[0]
                    confidence = count / data[data[feature] == value].shape[0]
                    if support >= min_sup and confidence >= min_conf:
                        condition = [{feature: value}]
                        frequent_itemsets.append({'condition': condition, 'class': class_value,
                                                  'support': support, 'confidence': confidence})
        return frequent_itemsets

    def calculate(self, itemset, data=None):
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        condition = itemset['condition']
        class_value = itemset['class']

        condition_condition = data[list(condition[0].keys())[0]] == list(condition[0].values())[0]
        for cond in condition[1:]:
            condition_condition &= (data[list(cond.keys())[0]] == list(cond.values())[0])

        condition_support_count = data[condition_condition].shape[0]
        condition_condition_with_class = condition_condition & (data['class'] == class_value)
        itemset_support_count = data[condition_condition_with_class].shape[0]

        if condition_support_count == 0:
            return 0, 0

        support = itemset_support_count / data.shape[0]
        confidence = itemset_support_count / condition_support_count
        return support, confidence

    def generate_frequents(self, freq_itemsets, k):
        min_sup = self.min_sup
        min_conf = self.min_conf
        frequents = []

        for i in range(len(freq_itemsets)):
            for j in range(i + 1, len(freq_itemsets)):
                if freq_itemsets[i]['class'] == freq_itemsets[j]['class']:
                    condest_a = freq_itemsets[i]['condition']
                    condest_b = freq_itemsets[j]['condition']
                    if len(set([list(cond.keys())[0] for cond in condest_a]) & set([list(cond.keys())[0] for cond in condest_b])) == k - 2:
                        new_condition = list({**{k: v for d in condest_a for k, v in d.items()},
                                              **{k: v for d in condest_b for k, v in d.items()}}.items())
                        if len(new_condition) == k:
                            condition = []
                            for cond in new_condition:
                                condition.append({cond[0]: cond[1]})
                            candidate = {'condition': condition, 'class': freq_itemsets[i]['class']}
                            sup, conf = self.calculate(candidate)
                            if sup > min_sup and conf > min_conf and conf > freq_itemsets[i]['confidence'] and conf > freq_itemsets[j]['confidence']:
                                candidate['support'] = sup
                                candidate['confidence'] = conf
                                frequents.append(candidate)
        return frequents

    def apriori(self):
        frequent_itemsets = []
        frequent_itemset = self.genFrequentOne()
        frequent_itemsets.extend(frequent_itemset)
        k = 2
        while True:
            frequent_itemset = self.generate_frequents(frequent_itemset, k)

            if len(frequent_itemset) == 0 or k >= 4:
                break
            else:
                k += 1
                frequent_itemsets.extend(frequent_itemset)

        return frequent_itemsets

    def prune(self, sorted_ruleitemset):
        data = self.data.copy()
        data_ = data.copy()
        rules = []
        default_class = data_['class'].value_counts().idxmax()

        while not data_.empty and sorted_ruleitemset:
            selected_rule = sorted_ruleitemset[0]
            cover_index = [index for index, row in data_.iterrows() if self.cover_for_prune(selected_rule, row)]

            if cover_index:
                data_ = data_.drop(cover_index, axis=0)
                rules.append(selected_rule)
                default_class = data_['class'].value_counts().idxmax() if data_.shape[0] != 0 else default_class

                sorted_ruleitemset_ = sorted_ruleitemset[1:]
                ruleitemset = []
                for ruleitem in sorted_ruleitemset_:
                    sup, conf = self.calculate(ruleitem, data_)
                    if sup > self.min_sup and conf > self.min_conf:
                        ruleitem['support'] = sup
                        ruleitem['confidence'] = conf
                        ruleitemset.append(ruleitem)
                if len(ruleitemset) != 0:
                    sorted_ruleitemset = sorted(ruleitemset, key=lambda x: (
                        x['confidence'], x['support'], len(x['condition'])), reverse=True)
                else:
                    break
            else:
                break

        return rules, default_class

    def cover_for_prune(self, ruleitem, instance):
        condition = ruleitem['condition']
        class_value = ruleitem['class']

        return all(instance[list(cond.keys())[0]] == list(cond.values())[0] for cond in condition) and instance['class'] == class_value

    def cover_for_predict(self, ruleitem, instance):
        condition = ruleitem['condition']
        return all(instance[list(cond.keys())[0]] == list(cond.values())[0] for cond in condition)

    def predict(self, data, rules, default_class=None):
        predict_y = []
        class_group = {}

        # 預測每一行的類別
        for _, row in data.iterrows():
            for rule in rules:
                if self.cover_for_predict(rule, row):
                    if rule['class'] in class_group:
                        class_group[rule['class']] += 1
                    else:
                        class_group[rule['class']] = 1
            if len(class_group) == 0:
                predict_y.append(default_class)
            else:
                predict_y.append(max(class_group, key=class_group.get))

        return predict_y
