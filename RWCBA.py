class RWCBA:
    def __init__(self, data):
        self.data = data

    def genFrequentOne(self, data, min_sup=0.1, min_conf=0.5):
        frequent_itemsets = []
        features = data.columns[:-1]
        class_values = data['class'].unique()
        for feature in features:
            for class_value in class_values:
                feature_values = data[feature].unique()
                for value in feature_values:
                    count = data[(data[feature] == value) & (data['class'] == class_value)].shape[0]
                    support = count / data.shape[0]
                    confidence = count / data[data[feature] == value].shape[0]
                    if support >= min_sup and confidence >= min_conf:
                        condition = [{feature: value}]
                        frequent_itemsets.append({'condition': condition, 'class': class_value,
                                                  'support': support, 'confidence': confidence})
        return frequent_itemsets

    def calculate(self, data, itemset):
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
                            sup, conf = self.calculate(self.data, candidate)
                            if sup > min_sup and conf > min_conf and conf > freq_itemsets[i]['confidence'] and conf > freq_itemsets[j]['confidence']:
                                candidate['support'] = sup
                                candidate['confidence'] = conf
                                frequents.append(candidate)
        return frequents

    def apriori(self, data, min_sup=0.01, min_conf=0.5):
        frequent_itemsets = []
        frequent_itemset = self.genFrequentOne(data, min_sup, min_conf)
        frequent_itemsets.extend(frequent_itemset)
        k = 2
        while True:
            frequent_itemset = self.generate_frequents(frequent_itemset, k)

            if len(frequent_itemset) == 0:
                break
            else:
                k += 1
                frequent_itemsets.extend(frequent_itemset)
        return frequent_itemsets

    def prune(self, data, sorted_ruleitemset):
        data_ = data.copy()
        data_y = data['class']
        rules = []

        for ruleitem in sorted_ruleitemset:
            # 找到符合當前規則的索引
            cover_index = [index for index, row in data_.iterrows() if self.cover_for_prune(ruleitem, row)]

            if cover_index:
                # 刪除符合當前規則的資料行
                data_ = data_.drop(cover_index, axis=0)
                rules.append(ruleitem)

                # 確定預設類別
                default_class = data_['class'].value_counts().idxmax()
                y = self.predict(data, rules, default_class)

                # 計算錯誤率
                error = sum(data_y[i] != y[i] for i in range(len(data_y)))

                ruleitem['error'] = error
                ruleitem['default_class'] = default_class

                rules[-1] = ruleitem

            if data_.empty:
                break

        # 找到最小錯誤率的規則
        min_error_rule = min(rules, key=lambda x: x['error'])
        min_error_index = rules.index(min_error_rule)
        default_class = min_error_rule['default_class']

        # 返回修剪後的規則和預設類別
        rules = rules[:min_error_index + 1]
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

        # 預測每一行的類別
        for _, row in data.iterrows():
            # 使用 next 和生成器表達式找到符合條件的第一個規則
            predicted_class = next((rule['class']
                                   for rule in rules if self.cover_for_predict(rule, row)), default_class)
            predict_y.append(predicted_class)

        return predict_y
