import sklearn.metrics as metrics
import warnings

warnings.filterwarnings("ignore")


class Validation():
    def __init__(self, data, rules, default, type='single', spare_rules=[]):
        self.data = data
        self.rules = rules
        self.default = default
        self.answer = list(self.data['class'])
        if (type == 'single'):
            self.forecast_result = self.predict()
        else:
            if len(spare_rules) != 0:
                self.spare_rules = spare_rules
                self.forecast_result = self.predict_apr()
            else:
                self.forecast_result = self.predict_group()
        self.report = self.get_classification_report()

    def predict(self):
        """
        單一預測
        """
        forecast_result = self.data.drop(columns=['class'], axis=1)
        for index in forecast_result.index:
            check = None
            for rule in self.rules:
                check = True
                conditions = rule.conditions
                for condition in conditions:
                    if forecast_result.loc[index, condition] != conditions[condition]:
                        check = False
                        break
                if check:
                    forecast_result.loc[index, 'class'] = str(rule.class_)
                    break
            if not check:
                forecast_result.loc[index, 'class'] = str(self.default)
        return list(forecast_result['class'])

    def predict_apr(self):
        """
        投票預測
        """
        data_class = self.data['class'].unique()
        forecast_result = self.data.drop(columns=['class'], axis=1)
        for index in forecast_result.index:
            _class = {}
            for i in data_class:
                _class[str(i)] = 0
            check = None
            for rule in self.rules:
                check = True
                conditions = rule.conditions
                for condition in conditions:
                    if forecast_result.loc[index, condition] != conditions[condition]:
                        check = False
                        break
                if check:
                    _class[str(rule.class_)] += 1
            if not check:
                for rule in self.spare_rules:
                    check = True
                    conditions = rule.conditions
                    for condition in conditions:
                        if forecast_result.loc[index, condition] != conditions[condition]:
                            check = False
                            break
                    if check:
                        _class[str(rule.class_)] += 1
                        break
            if not check:
                _class[self.default] += 1
            forecast_result.loc[index, 'class'] = str(max(
                _class, key=_class.get))
        return list(forecast_result['class'])

    def predict_group(self):
        """
        投票預測
        """
        data_class = self.data['class'].unique()

        forecast_result = self.data.drop(columns=['class'], axis=1)
        for index in forecast_result.index:
            _class = {}
            for i in data_class:
                _class[str(i)] = 0
            check = None
            for rule in self.rules:
                check = True
                conditions = rule.conditions
                for condition in conditions:
                    if forecast_result.loc[index, condition] != conditions[condition]:
                        check = False
                        break
                if check:
                    _class[str(rule.class_)] += 1
            if not check:
                _class[self.default] += 1
            forecast_result.loc[index, 'class'] = str(max(
                _class, key=_class.get))
        return list(forecast_result['class'])

    def get_classification_report(self):
        y_test = [str(i) for i in self.answer]
        y_pred = [str(i) for i in self.forecast_result]

        classification_report_ = metrics.classification_report(y_test, y_pred)

        print(classification_report_)

        # macro_f1 = metrics.f1_score(y_test, y_pred, average='macro')
        # accuracy = metrics.accuracy_score(y_test, y_pred)

        # print("Macro F1 Score:", macro_f1)
        # print("Accuracy:", accuracy)
