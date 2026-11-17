import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path):
    print("Loading dataset...")
    # Dataset load kar rahe hain (encoding latin-1 special characters ke liye)
    df = pd.read_csv(file_path, encoding='latin-1')
    print(f"Original dataset shape: {df.shape}")

    # 1. Irrelevant columns drop karna
    columns_to_drop = [
        'Customer Email', 'Customer Password', 'Product Image', 
        'Customer Street', 'Customer Fname', 'Customer Lname', 
        'Latitude', 'Longitude', 'Product Description'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # 2. Date columns ko datetime format mein convert karna
    date_columns = ['order date (DateOrders)', 'shipping date (DateOrders)']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3. Delivery Delay feature nikalna
    if 'order date (DateOrders)' in df.columns and 'shipping date (DateOrders)' in df.columns:
        df['Delivery_Delay_Days'] = (df['shipping date (DateOrders)'] - df['order date (DateOrders)']).dt.days

    # 4. Missing values handle karna
    print("Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    print(f"Preprocessed dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    path = os.path.join("data", "DataCoSupplyChainDataset.csv")
    if os.path.exists(path):
        cleaned_df = load_and_preprocess_data(path)
        
        # Cleaned data ko data/ folder mein save karna
        output_path = os.path.join("data", "cleaned_supply_chain.csv")
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned dataset successfully saved to {output_path}")
    else:
        print(f"Error: Dataset not found at {path}. Please place 'DataCoSupplyChainDataset.csv' inside the 'data/' folder.")