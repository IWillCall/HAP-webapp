from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,  FunctionTransformer, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import  MissingIndicator

from sklearn.base import BaseEstimator, TransformerMixin
import re
import dill
import numpy as np
import pandas as pd
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.statistics_ = {}
        self.numeric_columns_ = None
        self.non_numeric_columns_ = None

    def fit(self, X, y=None):
        self.statistics_ = {}
        self.numeric_columns_ = X.select_dtypes(include=["float32", "float64", "float16"]).columns
        self.non_numeric_columns_ = X.select_dtypes(exclude=["float32", "float64", "float16"]).columns

        for col in self.numeric_columns_:
            self.statistics_[col] = X[col].mean()

        for col in self.non_numeric_columns_:
            mode_value = X[col].mode()
            self.statistics_[col] = mode_value.iloc[0] if not mode_value.empty else None

        return self

    def transform(self, X):
        numeric_data = X[self.numeric_columns_].to_numpy(copy=True).astype(np.float32)
        non_numeric_data = X[self.non_numeric_columns_]
        if np.isnan(numeric_data).any():
            for i, col in enumerate(self.numeric_columns_):
                value = self.statistics_.get(col, None)
                mask = np.isnan(numeric_data[:, i])
                numeric_data[mask, i] = value

        numeric_df = pd.DataFrame(numeric_data, columns=self.numeric_columns_, index=X.index)
        non_numeric_data = non_numeric_data.fillna(self.statistics_)

        result = pd.concat([numeric_df, non_numeric_data], axis=1)
        return result[self.numeric_columns_.tolist() + self.non_numeric_columns_.tolist()]

    def set_output(self, transform=None):
        return self
        
    def __call__(self, X, y):
        return self.transform(X, y)
        
def format_age(row, **kwargs):
        if type(row) == float:
            return row
    
        replacements = {
            'Age': '',
            'to': '-',
            'or older': '+',
            ' ': ''
        }
        pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
        
        formated = pattern.sub(lambda match: replacements[match.group(0)], row)
        return formated
    
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        self.mappings = {key:value for key, value in self.mappings.items() if key in X.columns.to_list()}
        self.inverse_mappings = {col: {v: k for k, v in mapping.items()} for col, mapping in self.mappings.items()}

        return self 

    def transform(self, X):
        X_transformed = X.copy()
        for col, mapping in self.mappings.items():
            X_transformed[col] = X_transformed[col].map(mapping)
        return X_transformed

    def inverse_transform(self, X):
        X_inverse = X.copy()
        for col, inverse_mapping in self.inverse_mappings.items():
            X_inverse[col] = X_inverse[col].map(inverse_mapping)
        return X_inverse
    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else list(self.mappings.keys())
        
class Preprocessor:
    def __init__(self, is_bin_features=False):
        self.is_bin_features=is_bin_features
        self.bimodal_features = ["MentalHealthDays", "PhysicalHealthDays"]
        self.bins = [0, 3, 7, 14, 29, float("inf")]
        self.labels = ["0-3", "4-7", "8-14", "15-29", "30"]
        self.ordinal_mappings = {
        "GeneralHealth": {
            'Poor': 0,
            'Fair': 1, 
            'Good': 2, 
            'Very good': 3,  
            'Excellent': 4,
        },
        "AgeCategory":{
            '80+': 82.5, 
            '75-79': 77,
            '70-74': 72,
            '65-69': 67,
            '60-64': 62,
            '55-59': 57,
            '50-54': 52,
            '45-49': 46.5,
            '40-44': 42,
            '35-39': 37,
            '30-34': 32,
            '25-29': 27,
            '18-24': 21,
        }, 
        "RemovedTeeth": {
            'None of them': 0,
            '1 to 5': 1,
            '6 or more, but not all': 2,
            'All': 3
        },
        'LastCheckupTime': {
            'Within past year (anytime less than 12 months ago)': 0,
            'Within past 2 years (1 year but less than 2 years ago)': 1,
            'Within past 5 years (2 years but less than 5 years ago)': 2,
            '5 or more years ago': 3
        }
    }

    def bin_features(self, X):
        for feature in self.bimodal_features:
            X[feature] = pd.cut(X[feature], bins=self.bins, labels=self.labels, right=False)
            
    def create_transformers(self, num_features, binary_features, multiclass_features, ordinal_features, is_poly=False, add_missing_indicators=False):
        transformers = [] 
        if add_missing_indicators:
            all_featres = list(set(num_features + binary_features + multiclass_features + ordinal_features))
            transformers.append(("missing_indicators", MissingIndicator(), all_featres ))

        num_steps = [("imputer", CustomImputer())]
        if is_poly:
            num_steps.append(("poly", PolynomialFeatures(degree=2)))
        num_steps.append(("log", FunctionTransformer(lambda x: np.log1p(x))))
        num_steps.append(("scaler", MinMaxScaler()))
        num_pipe = Pipeline(num_steps)
        
        transformers.append(("num", num_pipe, num_features))
        binary_pipe = Pipeline([
            ("imputer", CustomImputer()),
            ("encoder", OneHotEncoder(sparse_output=False))
        ])
        transformers.append(("binary", binary_pipe, binary_features))
        multiclass_pipe = Pipeline([
            ("imputer", CustomImputer()),
            ("encoder", OneHotEncoder(sparse_output=False))
        ])
        if ordinal_features:
            multiclass_features = [feature for feature in multiclass_features if feature not in ordinal_features]
        
        transformers.append(("multiclass", multiclass_pipe, multiclass_features))
        if ordinal_features:
            ordinal_pipe = Pipeline([
                ("imputer", CustomImputer()),
                ("encoder", CustomOrdinalEncoder(self.ordinal_mappings))
            ])
            transformers.append(("ordinal", ordinal_pipe, ordinal_features))
        self.preprocessor = ColumnTransformer(transformers, remainder='drop', verbose=True)

        self.preprocessor.set_output(transform='pandas')
    def fit(self, X, y):
        if self.is_bin_features:
            X = self.bin_features(X)
            print("binned")
        X["AgeCategory"] = X["AgeCategory"].apply(format_age, axis=1)
        
        self.preprocessor.fit(X, y)
    def transform(self, X, y=None, add_missing_indicators=False):
        if self.is_bin_features:
            X = self.bin_features(X)
        
        X["AgeCategory"] = X["AgeCategory"].apply(format_age, axis=1)
        X = self.preprocessor.transform(X)
        X.columns = X.columns.str.replace(r'[^\w]', '_', regex=True)
        if y is None:
            return X
        else:
            y = y.map({"Yes": 1, "No": 0}).fillna(y)
            return X, y

        


def get_preprocessor():
    with open("utils/preprocessor_sl.pkl", "rb") as f:
        preprocessor_sl = dill.load(f)
    preprocessor = Preprocessor()
    # num_features = preprocessor_sl['num_features']
    # binary_features = preprocessor_sl['binary_features']
    # multiclass_features = preprocessor_sl['multiclass_features']
    # ordinal_features = preprocessor_sl['ordinal_features']
    # preprocessor.create_transformers(num_features=num_features,
    #                                 binary_features=binary_features,
    #                                 multiclass_features=multiclass_features,
    #                                 ordinal_features=ordinal_features)
    preprocessor.preprocessor = preprocessor_sl['preprocessor']

    return preprocessor

