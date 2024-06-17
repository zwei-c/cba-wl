from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


class CBA_WL:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.X_train, self.X_test, self.y_train, self.y_test, self.train, self.test = self.loadData()

    def loadData(self):
        dataset = fetch_ucirepo(id=self.dataset_id)
        features = dataset.data.features
        target = dataset.data.targets

    def randomForestFeatureImportances(self):
        print('Random Forest Feature Importances')

    def oneHotEncoding(self, df):
        # 將特徵轉換為所需格式，排除 'target' 列
        features_list = []
        for index, row in df.iterrows():
            features = []
            for col, value in row.items():
                features.append(f"{col}: {value}")
            features_list.append(features)
        # 使用 TransactionEncoder 進行編碼
        te = TransactionEncoder()
        te_ary = te.fit(features_list).transform(features_list)
        # 將編碼後的數據轉換為 DataFrame
        te_df = pd.DataFrame(te_ary, columns=te.columns_)


if __name__ == '__main__':
    dataset_dict = {
        'Car Evaluation': 19
    }
    cba = CBA_WL(19)
    # cba.randomForestFeatureImportances()
