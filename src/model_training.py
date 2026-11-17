import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    print("Loading cleaned dataset for model training...")
    file_path = os.path.join("data", "cleaned_supply_chain.csv")
    
    if not os.path.exists(file_path):
        print("Error: cleaned_supply_chain.csv not found. Please run preprocessing.py first.")
        return

    df = pd.read_csv(file_path)

    # Target variable aur features define karna
    target_column = 'Late_delivery_risk'
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in dataset.")
        return

    print("Preparing features and target...")
    # Modeling ke liye kuch important numeric aur categorical columns select karte hain
    # (Memory aur performance balance rakhne ke liye relevant columns lete hain)
    selected_features = [
        'Days for shipping (real)', 'Days for shipment (scheduled)', 
        'Benefit per order', 'Sales per customer', 'Order Item Quantity', 
        'Order Item Total', 'Product Price', 'Delivery_Delay_Days'
    ]
    
    # Check karna ki saare features dataframe mein hain ya nahi
    available_features = [col for col in selected_features if col in df.columns]
    
    X = df[available_features]
    y = df[target_column]

    # Handle any missing values if left
    X = X.fillna(X.median())

    # Train-Test Split (80% training, 20% testing)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier train karna
    print("Training Random Forest Classifier (this may take a minute)...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Model Evaluation
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Trained model ko save karna
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "supply_chain_model.pkl")
    joblib.dump(model, model_path)
    print(f"Trained model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()