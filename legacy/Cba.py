import pandas as pd


class Cba():
    def __init__(self, data: pd.DataFrame, rules: list, min_support_threshold: float = 0.01, min_confidence_threshold: float = 0.1):
        self.data = data
        self.rules = rules
        self.final_rules = []
        self.final_rules_ = []
        self.default = str(self.data['class'].mode()[0])
        self.min_support_threshold = min_support_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.strong_rules = []
        self.spare_rules = []

    def calculate_support_confidence(self, rule, data):
        if data.empty:
            return {"support": 0, "confidence": 0}

        conditions = rule.conditions
        support_count = 0
        confidence_count = 0

        for index, row in data.iterrows():
            if all(row[key] == value for key, value in conditions.items()):
                support_count += 1
                if row['class'] == rule.class_:
                    confidence_count += 1

        support_value = support_count / len(data) if len(data) > 0 else 0
        confidence_value = confidence_count / support_count if support_count > 0 else 0

        return {"support": support_value, "confidence": confidence_value}

    def cover(self):
        data = self.data
        for rule in self.rules:
            if data.shape[0] == 0:
                break
            conditions = rule.conditions
            data_frame_conditions = None

            for key in conditions:
                if (data_frame_conditions is None):
                    data_frame_conditions = data[key] == conditions[key]
                else:
                    data_frame_conditions &= data[key] == conditions[key]

            if data[data_frame_conditions].shape[0] > 0:
                data = data[~data_frame_conditions]
                if data.shape[0] != 0:
                    self.default = str(data['class'].mode()[0])
                self.final_rules.append({'rule': rule, 'error': 0})
                self.compare()
            else:
                continue
        # 找到錯誤數最少的規則 刪除他之後的規則
        error = 1000000000
        error_index = None
        for index in range(len(self.final_rules)):
            if self.final_rules[index]['error'] < error:
                error = self.final_rules[index]['error']
                error_index = index

        if (error_index == None):
            error_index = 0
        self.final_rules = self.final_rules[:error_index]

        for rule in self.final_rules:
            self.final_rules_.append(rule['rule'])

    def compare(self):

        def compare_sub(item, rule):
            conditions = rule.conditions
            for key in conditions:
                if item[key] != conditions[key]:
                    return False
            return True

        def get_errors(data, test_data):
            error = 0
            for index in range(data.shape[0]):
                if data.loc[index, 'class'] != test_data.loc[index, 'class']:
                    error += 1
            return error

        test_data = self.data.drop(columns=['class'], axis=1)

        for item in test_data.iterrows():
            check = False
            for rule in self.final_rules:
                if compare_sub(item[1], rule['rule']):
                    test_data.loc[item[0], 'class'] = rule['rule'].class_
                    check = True
                    break
            if not check:
                test_data.loc[item[0], 'class'] = self.default

        error = get_errors(self.data, test_data)
        self.final_rules[-1]['error'] = error

    def apr_cover(self):
        data = self.data
        self.strong_rules = []  # 用於存儲有效的規則
        self.spare_rules = []  # 用於存儲不再適用的規則

        while data.shape[0] > 0 and self.rules:
            # 根據支持度和置信度排序規則
            self.rules.sort(key=lambda x: (
                x.confidence, x.support), reverse=True)

            # 取出排名最高的規則
            rule = self.rules.pop(0)

            # 檢查規則是否在數據集中有效
            conditions = rule.conditions
            data_frame_conditions = None
            for key in conditions:
                if data_frame_conditions is None:
                    data_frame_conditions = data[key] == conditions[key]
                else:
                    data_frame_conditions &= data[key] == conditions[key]

            if data[data_frame_conditions].shape[0] > 0:
                # 如果規則有效，將其添加到 strong_rules
                self.strong_rules.append(rule)
                # 更新數據集，移除被規則覆蓋的數據
                data = data[~data_frame_conditions]
            else:
                # 如果規則不再有效，將其添加到 spare_rules
                self.spare_rules.append(rule)

            # 更新剩餘規則的支持度和置信度
            for r in self.rules:
                metrics = self.calculate_support_confidence(r, data)
                r.support = metrics["support"]
                r.confidence = metrics["confidence"]

            # 篩選出不滿足閾值的規則
            to_spare_rules = [r for r in self.rules if r.support <
                              self.min_support_threshold or r.confidence < self.min_confidence_threshold]
            self.spare_rules.extend(to_spare_rules)

            self.rules = [r for r in self.rules if r.support >=
                          self.min_support_threshold and r.confidence >= self.min_confidence_threshold]
