import pandas as pd

from sklearn.preprocessing import MinMaxScaler

def hybrid_recommends(cf_predictions, cbf_predictions):
    # Step 5: Combine CF and CBF Scores
    # Normalize predictions for each method
    scaler = MinMaxScaler()
    cf_normalized = scaler.fit_transform(cf_predictions)
    cbf_normalized = scaler.fit_transform(cbf_predictions)

    # Weighted combination
    alpha = 0.5  # Weight for CF; 1-alpha for CBF
    hybrid_scores = alpha * cf_normalized + (1 - alpha) * cbf_normalized
    
    return hybrid_scores