class RWCBA:
    def __init__(self, data, importance=[]):
        self.data = data
        self.importance = importance

    def genFrequentOne(self, min_sup=0.1, min_conf=0.5):
        data = self.data.copy()
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
                    support *= self.importance[feature]
                    confidence = count / data[data[feature] == value].shape[0]
                    if support >= min_sup and confidence >= min_conf:
                        condition = [{feature: value}]
                        hm = 2 * support * confidence / (support + confidence)
                        frequent_itemsets.append({'condition': condition, 'class': class_value,
                                                  'support': support, 'confidence': confidence, 'hm': hm})
        return frequent_itemsets

    def calculateConditionWeight(self, condition):
        importance = self.importance
        condition_len = len(condition)
        weight = 0
        for cond in condition:
            weight += importance[list(cond.keys())[0]]
        return weight / condition_len

    def calculate(self, itemset):
        data = self.data.copy()
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
        weight_support = support * self.calculateConditionWeight(condition)
        confidence = itemset_support_count / condition_support_count
        return weight_support, confidence

    def generate_frequents(self, freq_itemsets, k, min_sup=0.01, min_conf=0.5):
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
                            weight_support, conf = self.calculate(candidate)
                            if weight_support > min_sup and conf > min_conf and conf > freq_itemsets[i]['confidence'] and conf > freq_itemsets[j]['confidence']:
                                candidate['support'] = weight_support
                                candidate['confidence'] = conf
                                candidate['hm'] = 2 * candidate['support'] * candidate['confidence'] / \
                                    (candidate['support'] + candidate['confidence'])
                                frequents.append(candidate)
        return frequents

    def apriori(self, min_sup=0.01, min_conf=0.5):
        frequent_itemsets = []
        frequent_itemset = self.genFrequentOne(min_sup, min_conf)
        frequent_itemsets.extend(frequent_itemset)
        k = 2
        while True:
            frequent_itemset = self.generate_frequents(frequent_itemset, k, min_sup, min_conf)

            if len(frequent_itemset) == 0:
                break
            else:
                k += 1
                frequent_itemsets.extend(frequent_itemset)
        return frequent_itemsets

    def prune(self, sorted_ruleitemset):
        data = self.data.copy()
        data_ = data.copy()
        data_y = data['class']
        rules = []
        strong_rules = []
        spare_rules = []
        for ruleitem in sorted_ruleitemset:
            # 找到符合當前規則的索引
            cover_index = [index for index, row in data_.iterrows() if self.cover_for_prune(ruleitem, row)]

            if cover_index:
                # 刪除符合當前規則的資料行
                data_ = data_.drop(cover_index, axis=0)
                rules.append(ruleitem)

                # 確定預設類別
                default_class = data_['class'].value_counts().idxmax()
                y = self.predict_for_prune(rules, default_class)

                # 計算錯誤率
                error = sum(data_y[i] != y[i] for i in range(len(data_y)))

                ruleitem['error'] = error
                ruleitem['default_class'] = default_class

                rules[-1] = ruleitem
            else:
                spare_rules.append(ruleitem)

            if data_.empty:
                break

        # 找到最小錯誤率的規則
        min_error_rule = min(rules, key=lambda x: x['error'])
        min_error_index = rules.index(min_error_rule)
        default_class = min_error_rule['default_class']

        # 返回修剪後的規則和預設類別
        strong_rules = rules[:min_error_index + 1]
        spare_rules.extend(rules[min_error_index + 1:])
        spare_rules.sort(key=lambda x: x['hm'], reverse=True)
        return strong_rules, spare_rules, default_class

    def cover_for_prune(self, ruleitem, instance):
        condition = ruleitem['condition']
        class_value = ruleitem['class']

        return all(instance[list(cond.keys())[0]] == list(cond.values())[0] for cond in condition) and instance['class'] == class_value

    def cover_for_predict(self, ruleitem, instance):
        condition = ruleitem['condition']
        return all(instance[list(cond.keys())[0]] == list(cond.values())[0] for cond in condition)

    def predict_for_prune(self, rules,  default_class=None):
        data = self.data.copy()
        predict_y = []
        # 預測每一行的類別
        for _, row in data.iterrows():
            # 使用 next 和生成器表達式找到符合條件的第一個規則
            predicted_class = next((rule['class']
                                    for rule in rules if self.cover_for_predict(rule, row)), default_class)
            predict_y.append(predicted_class)

        return predict_y

    def predict(self, data, strong_rules, spare_rules, default_class=None):
        predict_y = []
        # 預測每一行的類別
        for _, row in data.iterrows():
            class_group = {}
            for rule in strong_rules:
                if self.cover_for_predict(rule, row):
                    class_value = rule['class']
                    if class_value in class_group.keys():
                        class_group[class_value].append(rule['hm'])
                    else:
                        class_group[class_value] = [rule['hm']]

            if not class_group:
                for rule in spare_rules:
                    if self.cover_for_predict(rule, row):
                        class_value = rule['class']
                        if class_value in class_group.keys():
                            class_group[class_value].append(rule['hm'])
                        else:
                            class_group[class_value] = [rule['hm']]

            if class_group:
                mean_hm = {k: sum(v) / len(v) for k, v in class_group.items()}
                predict_y.append(max(mean_hm, key=mean_hm.get))
            else:
                predict_y.append(default_class)

        return predict_y
