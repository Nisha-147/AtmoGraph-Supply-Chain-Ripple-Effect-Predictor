import pandas as pd
import matplotlib
matplotlib.use('Agg')  # <--- Yeh line add karni hai (import pyplot se pehle)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    print("Loading cleaned dataset for EDA...")
    file_path = os.path.join("data", "cleaned_supply_chain.csv")
    
    if not os.path.exists(file_path):
        print("Error: cleaned_supply_chain.csv not found. Please run preprocessing.py first.")
        return

    df = pd.read_csv(file_path)
    
    # Reports folder create karna charts save karne ke liye
    os.makedirs("reports", exist_ok=True)

    # 1. Late Delivery Risk Distribution Plot
    if 'Late_delivery_risk' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Late_delivery_risk', palette='Set2', data=df)
        plt.title('Distribution of Late Delivery Risk (0: No Risk, 1: Risk)')
        plt.xlabel('Late Delivery Risk')
        plt.ylabel('Count')
        plt.savefig('reports/late_delivery_distribution.png')
        plt.close()
        print("Saved: reports/late_delivery_distribution.png")

    # 2. Shipping Mode vs Delivery Delay
    if 'Shipping Mode' in df.columns and 'Delivery_Delay_Days' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.barplot(x='Shipping Mode', y='Delivery_Delay_Days', data=df, palette='viridis', ci=None)
        plt.title('Average Delivery Delay by Shipping Mode')
        plt.xlabel('Shipping Mode')
        plt.ylabel('Average Delay (Days)')
        plt.xticks(rotation=45)
        plt.savefig('reports/shipping_mode_delay.png')
        plt.close()
        print("Saved: reports/shipping_mode_delay.png")

    print("EDA completed successfully! Visualizations saved in 'reports/' folder.")

if __name__ == "__main__":
    perform_eda()