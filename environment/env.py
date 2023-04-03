from sklearn.preprocessing import PolynomialFeatures
import joblib


class env:
    def __init__(self, dataset) -> None:
        self.ds = dataset
        self.income_part = {
            'deterministic_model': joblib.load('./datasets/model/income.pkl'),
            'feature_trans': PolynomialFeatures(degree=3),
            'persistent_influence_parameter': 1.0005,
            'sigma_transitory': 0.05,
        }
    

    def evlove_income(self, ):
        return