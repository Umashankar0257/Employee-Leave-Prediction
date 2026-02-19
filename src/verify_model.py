import joblib
import numpy as np

def test_model():
    print("Testing model artifacts...")
    try:
        model = joblib.load("model.pkl")
        encoders = joblib.load("encoders.pkl")
        tfidf = joblib.load("tfidf.pkl")
        
        # Mock input
        # ['LEAVECODE', 'applied_day_of_week', 'applied_month', 'leave_days', 'ELBALANCE', 'HPLBALANCE', 'total_past_leaves']
        # Use string encoders for categorical
        enc_leave = encoders['LEAVECODE'].transform(['EL'])[0]
        enc_day = encoders['applied_day_of_week'].transform(['5.0'])[0]
        enc_month = encoders['applied_month'].transform(['10.0'])[0]
        
        numeric = [enc_leave, enc_day, enc_month, 5, 303.0, 263.0, 9]
        reason_tfidf = tfidf.transform(["Personal work and family function"]).toarray()
        
        X = np.hstack(([numeric], reason_tfidf))
        
        prob = model.predict_proba(X)[0][1]
        print(f"Prediction Probability: {prob:.4f}")
        
        assert 0.0 <= prob <= 1.0
        print("Test Passed: Probability is within range.")
        
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_model()
