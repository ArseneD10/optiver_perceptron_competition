from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier 

from torch import nn

# abstract class
class PredictionSystem:

     def __init__(self, model, cluster: int):
          self.model = model 
          self.cluster = cluster

     def fit():
          ...
     def predict():
          ...


class SkLearnPrediction(PredictionSystem):

     def __init__(self, model: LogisticRegression, cluster: int): # logistic for example but work for any other sklearn compatible model
          super().__init__(model, cluster)

     def fit(self, X, y):
          self.model.fit(X, y)

     def predict(self, X):
          return self.model.predict_proba(X)


class NeuralPrediction(PredictionSystem):
     
    def __init__(self, model: "NeuralModel", cluster: int):
          super().__init__(model, cluster)

    def fit(self):
        ...

    def predict():
       ...

class NeuralModel(nn.Module):
     
     def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)