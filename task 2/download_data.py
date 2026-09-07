"""
Download Customer Churn Dataset
"""

import os
import pandas as pd
import certifi
import ssl
import urllib.request

def download_customer_churn_dataset():
    """
    Download Customer Churn Dataset.
    """
    print("="*80)
    print("📥 DOWNLOADING CUSTOMER CHURN DATASET")
    print("="*80)
    
    # Direct download links (you may need to adjust these)
    urls = [
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    ]
    
    filename = "customer_churn.csv"
    
    for url in urls:
        try:
            print(f"\nAttempting to download from: {url}")
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(url, context=ssl_context) as response, open(filename, 'wb') as output_file:
                output_file.write(response.read())
            
            # Verify the file
            df = pd.read_csv(filename)
            print(f"✅ Dataset downloaded successfully!")
            print(f"📊 Shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"\n💾 Saved as: {filename}")
            
            # Display sample data
            print("\n📋 Sample data:")
            print(df.head())
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to download: {e}")
            continue
    
    print("\n❌ Automatic download failed.")
    print("\nPlease download manually from:")
    print("https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset")
    print("\nOr use the IBM Telco Customer Churn dataset:")
    print("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("\nSave the file as 'customer_churn.csv' in the project folder.")
    
    return False

if __name__ == "__main__":
    download_customer_churn_dataset()