import torch
from typing import Any
from itertools import product

def agg_mean(X: torch.Tensor):
    return X.mean(dim=-1, keepdim=True)

def agg_max(X: torch.Tensor):
    return X.max(dim=-1, keepdim=True).values

def agg_std(X: torch.Tensor):
    return X.std(dim=-1, keepdim=True)

def agg_min(X: torch.Tensor):
    return X.min(dim=-1, keepdim=True).values

def agg_quantile(X: torch.Tensor, quantile: float = 0.9):
    return torch.quantile(X, q=quantile, dim=-1, keepdim=True) 

class AggregationFunction:

    def __init__(self, 
                 function: str, 
                 col_index: int,
                 kwargs: dict[str, Any] = {}):
        self.name = function         
        self.function = self._str_to_function(function)
        self.kwargs = kwargs
        self.col_index = col_index

    def __call__(self, x):
        return self.function(x[: ,self.col_index], **self.kwargs)
    
    def __repr__(self):
        return f"{self.name}:index-{self.col_index}"
    
    def dump(self):
        return {
            "function": self.name,
            "col_index": self.col_index,
            "kwargs": self.kwargs
        }
    
    def _str_to_function(self, s: str):
        s = s.lower().replace(" ", "")
        if s == "max":
            return agg_max
        elif s == "min":
            return agg_min
        elif s == "mean":
            return agg_mean 
        elif s == "std":
            return agg_std
        else:
            raise ValueError("Name must be implemented in aggregation.py")   

class ManualAggregation:
    def __init__(self, agg_function: list[AggregationFunction]):
        self.aggf = agg_function
        self.N = len(agg_function)

    def __call__(self, X):
        # X axis must be Batch, Sequence, Features
        AXIS = 1
        shape = X.shape
        M = shape[-1]
        assert(len(shape) >= 3), f"X shape must be greater than 3, got {X.shape}"
        new_shape = list(shape)
        new_shape[AXIS] = 1
        out = torch.empty(size=new_shape, device=X.device)
        for i in range(M):
            out[..., i] = self.aggf[i](X)
        return out, None, None # same as VAE output
    

class AggregationCombinator:

    def __init__(self, available_function, n_col: int):
        self.availble_index = list(range(n_col))
        self.available_function = available_function
        
    def get_agg_function(self):
        available_function_twice = self.get_twice_index(self.available_function)
        candidates = list(product(available_function_twice, repeat=len(self.available_index)))

        return self.from_candidates(candidates)

    @staticmethod   
    def get_twice_index(index):
        out = []
        for i in range(len(index)):
            for j in range(i+1, len(index)):
                out.append((index[i], index[j]))

        return out
    @staticmethod
    def from_candidates(candidates):
        all_candidates = []
        for candidate in candidates:
            candidate_function = []
            for i, functions in enumerate(candidate):
                for function in functions:
                    agg_function = AggregationFunction(function, i)
                    candidate_function.append(agg_function)
            all_candidates.append(candidate_function)
        return all_candidates
