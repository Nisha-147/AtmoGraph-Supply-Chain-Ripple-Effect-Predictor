import pandas as pd
import numpy as np

# 1. Load the dataset from the main folder
df = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')

# 2. Select only the necessary columns
selected_cols = [
    'Type', 'Days for shipping (real)', 'Days for shipment (scheduled)',
    'Benefit per order', 'Sales per customer', 'Delivery Status',
    'Late_delivery_risk', 'Category Name', 'Customer City', 'Customer Country',
    'Customer Segment', 'Department Name', 'Market', 'Order City',
    'Order Country', 'order date (DateOrders)', 'Order Item Quantity',
    'Order Item Total', 'Order Region', 'Order Status',
    'shipping date (DateOrders)', 'Shipping Mode'
]
df = df[selected_cols].copy()

# 3. Convert date columns to proper datetime format
df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
df['shipping_date'] = pd.to_datetime(df['shipping date (DateOrders)'])

# 4. Calculate shipping delay (difference in days)
df['shipping_delay_days'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

# 5. Create the Ripple Effect feature: Rolling average delay per region based on time order
df = df.sort_values(by=['Order Region', 'order_date']).reset_index(drop=True)
df['regional_rolling_delay'] = df.groupby('Order Region')['shipping_delay_days'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
).fillna(0)

print("Data successfully loaded and cleaned! Dataset shape:", df.shape)
print(df.head(5))