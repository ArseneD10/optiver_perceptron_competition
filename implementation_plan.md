Plan:

Part 1.
        -> Define set of agg function + (V)AE attention architecture, denote the transformation f
        -> Train the system f(x) -> Z -> Encoder(Z) -> recon -> normalized_loss

    Extract candidates with the lowest loss

-- Evaluation 1 -- 
    Cluster 'consistency' (AAABBB instead of ABABA)
    Silhouete score
    KS between distribution of cluster

        Not in evaluation but good to know:
            Volatility by cluster
            Return by cluster
            Autocorr & corr by cluster

Part 2.
        -> Feature segmentation (as in the paper) & regime denoising via causal argmax
        -> Linear model: alpha*Imbalance + beta*TradeImbalance + theta*CumReturn + epsilon -> Logistic regression on bins
        -> Deep Neural Model given Z -> Cross entropy on bins

--- Evaluation 2 --- 

    -> Fit baseline Model
    -> Compare baseline with trained system on informative metrics (spearman IC, Hit-ratio, log-loss) 
    -> Compare trained system with same system with permuted head (same metrics)


