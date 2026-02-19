import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Constants
DATA_PATH = "CLEANED_DATASET.csv"
MODEL_PATH = "model.pkl"
ENCODERS_PATH = "encoders.pkl"
TFIDF_PATH = "tfidf.pkl"

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Fill missing values if any
    df = df.fillna(0) # Simplified filling, ideally checking columns first
    
    # Encoders dict to save later
    encoders = {}
    
    # Categorical columns to encode
    cat_cols = ['LEAVECODE', 'applied_day_of_week', 'applied_month']
    
    for col in cat_cols:
        le = LabelEncoder()
        # Convert to string to ensure consistent encoding
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    return df, encoders

def train_model(df, encoders):
    print("Vectorizing LVREASON...")
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    X_reason = tfidf.fit_transform(df['LVREASON'].fillna('')).toarray()
    
    # Features
    # Combining numerical/categorical features with TF-IDF features
    # Adjust feature list based on dataset inspection
    feature_cols = ['LEAVECODE', 'applied_day_of_week', 'applied_month', 
                    'leave_days', 'ELBALANCE', 'HPLBALANCE', 'total_past_leaves']
    
    X_numeric = df[feature_cols].values
    X = np.hstack((X_numeric, X_reason))
    y = df['approval_status']
    
    print(f"Training shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    return clf, tfidf

def main():
    try:
        df = load_data(DATA_PATH)
        df, encoders = preprocess_data(df)
        model, tfidf = train_model(df, encoders)
        
        print("Saving artifacts...")
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        joblib.dump(tfidf, TFIDF_PATH)
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
