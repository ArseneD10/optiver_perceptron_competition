import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kstest


def argmax_filter(sequences: np.ndarray, kernel_size: int = 10):
    #window = sliding_window_view(sequences, window_shape=kernel_size, axis=0)
    # non vectorized:
    out = []
    for w in range(kernel_size, len(sequences)):
        window = sequences[w-kernel_size: w]
        count = np.unique_counts(window)
        argmax = np.argmax(count.counts)
        value = count.values[argmax]
        out.append(value)
    return np.array(out)

def calibrate_dispersion(centroid: np.ndarray, labels: np.ndarray, latents: np.ndarray):
    unique_labels = np.unique(labels)
    vi = [0 for _ in range(len(unique_labels))]
    for label in unique_labels:
        ci = centroid[label]
        latent = latents[labels == label]
        di = ((ci - latent)**2).sum(axis=1)**0.5
        vi[label] = np.std(di)
    return vi

def get_centroid_distance(centroid: np.ndarray, latents: np.ndarray, vi: np.ndarray):
    dist = []
    for idx, ci in enumerate(centroid):
        di = ((ci - latents)**2).sum(axis=1)**0.5 / vi[idx] # [N, 1]
        dist.append(di)
    dist = np.stack(dist, axis=-1) # N, 2
    return np.min(dist, axis=-1)
    
def calibrate_tau(centroid: np.ndarray, latents: np.ndarray, vi: np.ndarray, quantile: float= 0.99):
    # variable stats on
    # proba of not being part of the distribution of cluster distance 
    dist = get_centroid_distance(centroid, latents, vi)
    tau = np.quantile(dist, q=quantile)
    return tau


class FilterSystem:

    def __init__(self, reference_cluster: np.ndarray, reference_latents: np.ndarray, centroid: np.ndarray, window: int):
        self.centroid = centroid
        self.vi = calibrate_dispersion(centroid=centroid, labels=reference_cluster, latents=reference_latents)
        self.si = get_centroid_distance(centroid, reference_latents, self.vi)
        self.tau = calibrate_tau(centroid, reference_latents, self.vi)
    
    def rolling_metrics(self, latents_window: np.ndarray):
        window_si =  get_centroid_distance(self.centroid, latents_window, self.vi)
        ks_score = kstest(self.si, window_si)
        novelty = window_si[-1] > self.tau # any or last depending on the usage; default last
        return {"ks_score": ks_score, "novelty": novelty}


