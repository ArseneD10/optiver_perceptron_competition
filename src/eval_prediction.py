from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from scipy.stats import spearmanr
from src.prediction import PredictionSystem
import numpy as np
import matplotlib.pyplot as plt

class PredictionEvaluator:

    def __init__(self, model: PredictionSystem, X: np.ndarray, y: np.ndarray):
        self.model = model
        self.pred_proba = model.predict(X)
        self.pred = np.argmax(self.pred_proba, axis=1).astype(int)
        self.y = y

    def log_loss(self): # equivalent to cross-entropy
        p_pred = []
        for p, p_true in zip(self.pred_proba, self.y):
            p_pred.append(p[p_true])
        p_pred = np.stack(p_pred)
        return -np.mean(np.log(p_pred))

    def hit_ratio(self):
        return accuracy_score(self.y, self.pred)

    def ic_ratio(self, mode: str = "spearman"):
        # Since there is order between class (class 0 > class 1 > class 2) we can can use IC as informative metrics
        if mode == "spearman":
            return spearmanr(self.pred, self.y)
        else:
            return np.corrcoef(self.pred, self.y)[0, 1]

    def confusion_matrix(self, plot: bool = True):
        labels = ["negative", "neutral", "positive"]
        cm = confusion_matrix(self.y, self.pred)
        if plot:
            fig, ax = plt.subplots(figsize=(12, 12))
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=ax)
            plt.show()
        return cm
    

    def evaluate(self, ic_mode: str = "spearman", plot: bool = True):
        log_loss = self.log_loss()
        hr = self.hit_ratio()
        ic = self.ic_ratio(mode=ic_mode)
        cm = self.confusion_matrix(plot=plot)

        return {
            "log_loss": log_loss,
            "hit_ratio": hr,
            "ic": ic,
            "cm": cm
        }

    def cmp(self, rhs: "PredictionEvaluator") -> "PredictionMetric":
        return PredictionMetric(**self.evaluate()) - PredictionMetric(**rhs.evaluate())

class PredictionMetric:
    
    def __init__(self, log_loss, hit_ratio, ic, cm):
        self.log_loss = log_loss
        self.hit_ratio = hit_ratio
        self.ic = ic 
        self.cm = cm 

    def __add__(self, rhs: "PredictionMetric") -> "PredictionMetric":
        return PredictionMetric(
            log_loss=self.log_loss + rhs.log_loss,
            hit_ratio=self.hit_ratio+rhs.hit_ratio,
            ic = self.ic + rhs.ic,
            cm = self.cm + rhs.cm
        )