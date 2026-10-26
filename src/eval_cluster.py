import numpy as np 
from scipy.stats import kstest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class ClusterEvaluator:

    def __init__(self, latents_seq: np.ndarray, k_samples = [2, 3, 4]):
        self.latents = latents_seq
        self.k_samples = k_samples
        self.k_means = self._get_kmean()

    def _get_kmean(self) -> dict:
        kmeans = {}
        for k in self.k_samples:
            m = KMeans(n_clusters=k, max_iter=1000, random_state=22)
            cluster = m.fit_predict(self.latents)
            kmeans[k] = {"model": m, "cluster": cluster}

        return kmeans
    

    def eval_persistance(self):
        # turnover / max_turnver
        # since max_turnover is case where we have ABABA then the number is simply lenght/2
        # mean duration: 
        scores = {}

        for k in self.k_samples:
            sequences = self.k_means[k]["cluster"]
            size = len(sequences)
            turnover = (np.abs(np.diff(sequences)) >= 1).sum()
            max_turnover = size - 1
            score_turnover = turnover / max_turnover

            # @todo vectorize this algorithm
            mean_duration = 0.0
            current_duration = 1
            N = 0
            for i in range(1, len(sequences)):
                if sequences[i] == sequences[i-1]:
                    current_duration += 1
                else:
                    mean_duration += current_duration
                    current_duration = 1
                    N += 1
            mean_duration /= N     

            scores[k] = {"turnover": score_turnover, "mean_duration": mean_duration}   
        return scores
    
    def eval_silhouete_separation(self):
        scores = {}
        latent_reduced = PCA(n_components=2).fit_transform(self.latents)
        fig, ax = plt.subplots(figsize=(21, 9), ncols=len(self.k_samples))
        for i, k in enumerate(self.k_samples):
            cluster = self.k_means[k]["cluster"]
            kscore = silhouette_score(self.latents, cluster)
            scores[k] = kscore 
            ax[i].set_title(f"k-{k}")
            ax[i].scatter(latent_reduced[:, 0], latent_reduced[:, 1], c=cluster)
            ax[i].grid()
        plt.show()
    
        return scores
    
    def eval_regime_returns(self, returns: np.ndarray):
        results = {}
        for k in self.k_samples:
            clusters = self.k_means[k]["cluster"]
            fig, ax = plt.subplots(figsize=(21, 9), ncols=k)
            means = []
            std = [] 
            ks_test = []
            for ki in range(k):
                sample_return = returns[clusters == ki]
                other_return =returns[clusters != ki]
                mu = sample_return.mean()
                sigma = sample_return.std()
                means.append(mu)
                std.append(sigma)
                ax[ki].set_title(f"K={ki} Mu~{mu:.2f} | std~{sigma:.3f}")
                ax[ki].hist(sample_return, bins=20)
                ax[ki].axvline(mu, label="Expectation", c="black")
                ax[ki].grid()
                ax[ki].legend()
                ks_test.append(kstest(sample_return, other_return))
            results[k] = {"ks": ks_test, }
        plt.show()