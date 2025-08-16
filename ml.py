# ml.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

def train(csv_path=None, model_path="data/lead_model.pkl"):
    """Train clustering model on CRM data"""
    
    if csv_path is None:
        # Try to find default CRM data
        csv_path = "data/crm_data.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No CRM data found at {csv_path}")
    
    # Load CRM data
    df = pd.read_csv(csv_path)
    
    # Basic feature engineering (customize based on your CRM data structure)
    features = []
    
    # Try to extract common CRM features
    if 'email_opens' in df.columns:
        features.append('email_opens')
    if 'email_clicks' in df.columns:
        features.append('email_clicks')
    if 'meeting_booked' in df.columns:
        features.append('meeting_booked')
    if 'company_size' in df.columns:
        features.append('company_size')
    if 'industry' in df.columns:
        # Encode industry as numeric
        df['industry_encoded'] = pd.Categorical(df['industry']).codes
        features.append('industry_encoded')
    
    # If no specific features found, use numeric columns
    if not features:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = numeric_cols[:5]  # Use first 5 numeric columns
    
    if not features:
        raise ValueError("No suitable features found in CRM data")
    
    # Prepare features
    X = df[features].fillna(0)
    
    # Train clustering model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    n_clusters = min(5, len(X) // 10)  # At least 10 samples per cluster
    n_clusters = max(2, n_clusters)  # At least 2 clusters
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Train classifier for prediction
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_scaled, clusters)
    
    # Save models
    os.makedirs("data", exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            'scaler': scaler,
            'kmeans': kmeans,
            'classifier': classifier,
            'features': features,
            'n_clusters': n_clusters
        }, f)
    
    print(f"Trained model with {n_clusters} clusters using features: {features}")
    return model_path

def predict(csv_path, model_path="data/lead_model.pkl"):
    """Predict lead segments for new CRM data"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train() first.")
    
    # Load model
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    scaler = model_data['scaler']
    classifier = model_data['classifier']
    features = model_data['features']
    
    # Load new data
    df = pd.read_csv(csv_path)
    
    # Prepare features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}. Using zeros.")
        for feature in missing_features:
            df[feature] = 0
    
    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Predict clusters
    predictions = classifier.predict(X_scaled)
    
    # Add predictions to dataframe
    result_df = df.copy()
    result_df['predicted_segment'] = predictions
    result_df['segment_probability'] = np.max(classifier.predict_proba(X_scaled), axis=1)
    
    return result_df

def get_segment_insights(model_path="data/lead_model.pkl"):
    """Get insights about each segment"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train() first.")
    
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    kmeans = model_data['kmeans']
    features = model_data['features']
    
    # Get cluster centers
    centers = kmeans.cluster_centers_
    
    insights = []
    for i, center in enumerate(centers):
        segment_insight = {
            'segment': i,
            'characteristics': {}
        }
        
        for j, feature in enumerate(features):
            segment_insight['characteristics'][feature] = center[j]
        
        insights.append(segment_insight)
    
    return insights
