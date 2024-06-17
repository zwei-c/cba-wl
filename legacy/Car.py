from RuleItem import RuleItem, RuleItem_Weight
import pandas as pd


class Car():
    def __init__(self, data: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.1, min_lift: float = 0.0, weights=[]):
        self.data = data
        self.label = list(self.data['class'].unique())
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rule: list = []  # class: ruleitem
        self.weights = weights

    def generate_candidate(self, rule_temp):

        def join(ruleitem1: RuleItem, ruleitem2: RuleItem):
            if ruleitem1.class_ != ruleitem2.class_:
                return None

            condiset1 = set(ruleitem1.conditions)
            condiset2 = set(ruleitem2.conditions)

            if (condiset1 == condiset2):  # 判斷全部的條件屬性是否相等，不一樣表示有不同的條件屬性，可以繼續合併。
                return None

            intersect = condiset1 & condiset2  # 交集

            for item in intersect:  # 判斷交集是否相等，如果相等表示可以合併，不相等表示相同的條件屬性下部分屬性值不同，不可以合併。
                if ruleitem1.conditions[item] != ruleitem2.conditions[item]:
                    return None

            condiset = condiset1 | condiset2  # 合併兩個條件屬性集合

            # if len(condiset) > 5:
            #     return None

            new_item = {'conditions': {}, 'class': None}
            for item in condiset:
                if item in condiset1:
                    new_item['conditions'][item] = ruleitem1.conditions[item]
                else:
                    new_item['conditions'][item] = ruleitem2.conditions[item]
            new_item['class'] = ruleitem1.class_
            return new_item

        candidate = []
        for i in range(0, len(rule_temp)):
            for j in range(i+1, len(rule_temp)):
                new_item = join(rule_temp[i], rule_temp[j])
                if new_item != None:
                    if self.weights == []:
                        rule_item = RuleItem()
                    else:
                        rule_item = RuleItem_Weight(self.weights)
                    rule_item.set_ruleitem(
                        new_item, self.data, self.min_support, self.min_confidence, self.min_lift)
                    if rule_item.is_ruleitem:
                        if rule_item.confidence > rule_temp[i].confidence and rule_item.confidence > rule_temp[j].confidence:
                            candidate.append(rule_item)
        return candidate

    def generate_frequent(self):
        label = self.label
        data = self.data
        min_support = self.min_support
        min_confidence = self.min_confidence
        min_lift = self.min_lift
        rule_temp = []
        for i in data.columns[:-1]:
            for j in data[i].unique():
                for k in label:
                    if self.weights == []:
                        rule_item = RuleItem()
                    else:
                        rule_item = RuleItem_Weight(self.weights)
                    candidate = {'conditions': {i: j}, 'class': k}
                    if (rule_item.set_ruleitem(candidate, data, min_support, min_confidence, min_lift)):
                        self.rule.append(rule_item)
                        rule_temp.append(rule_item)
        while rule_temp != []:
            candidate = self.generate_candidate(rule_temp)
            rule_temp = []
            if candidate == []:
                break
            for item in candidate:
                self.rule.append(item)
                rule_temp.append(item)

    def sort_rule(self, type=1, hm=True):
        """
        sort_rule(type=1)
        type = 1: sort by confidence > support > lift
        type = 2: sort by hm(confidence, support, lift)
        type = 3: sort by hm(confidence, weights_support)
        """

        if type == 1:
            self.rule.sort(key=lambda x: (
                x.confidence, x.support, x.lift), reverse=True)
        elif type == 2:
            if hm:
                self.rule.sort(key=lambda x: (
                    3/(1/x.support + 1/x.confidence + 1/x.lift)), reverse=True)
            else:
                self.rule.sort(key=lambda x: (
                    (x.confidence + x.support + x.lift)/3), reverse=True)
        elif type == 3:
            if hm:
                self.rule.sort(key=lambda x: (
                    2/(1/x.confidence+1/x.weights_support)), reverse=True)
            else:
                self.rule.sort(key=lambda x: (
                    (x.confidence + x.weights_support)/2), reverse=True)
