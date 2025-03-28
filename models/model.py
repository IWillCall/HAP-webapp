from sklearn.linear_model import LogisticRegression
import dill

def get_model():
    with open("models/logreg_cholesky.pkl", "rb") as f:
      model = dill.load(f)
    return model