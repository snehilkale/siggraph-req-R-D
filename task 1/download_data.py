import urllib.request
import zipfile
import os
import ssl
import pandas as pd

def download_sms_spam_dataset():
    """
    Download the SMS Spam Collection Dataset from UCI repository
    """
    print("📥 Downloading SMS Spam Collection Dataset...")
    
    # URL for the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    
    try:
        # Download the file
        default_cafile = ssl.get_default_verify_paths().cafile
        cafile = default_cafile if default_cafile and os.path.isfile(default_cafile) else "/etc/ssl/cert.pem"
        ssl_context = ssl.create_default_context(cafile=cafile)
        with urllib.request.urlopen(url, context=ssl_context) as response, open(zip_path, "wb") as output_file:
            output_file.write(response.read())
        print("✅ Download complete!")
        
        # Extract the zip file
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Convert from TSV to CSV format
        if os.path.exists("SMSSpamCollection"):
            # Read the TSV file
            df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, 
                             names=['label', 'message'])
            
            # Save as CSV
            df.to_csv("spam.csv", index=False)
            
            # Remove the original TSV file
            os.remove("SMSSpamCollection")
            
            print("✅ Converted to spam.csv")
        
        # Clean up zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
        
        print("\n✅ Dataset downloaded and prepared successfully!")
        print(f"📁 File saved as: spam.csv")
        print(f"📊 File size: {os.path.getsize('spam.csv') / 1024:.2f} KB")
        
        # Display sample data
        print("\n📋 Sample data:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nAlternative download options:")
        print("1. Download from Kaggle: https://www.kaggle.com/uciml/sms-spam-collection-dataset")
        print("2. Download from UCI: https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
        print("3. Manually download and save as 'spam.csv' with columns: label,message")

if __name__ == "__main__":
    download_sms_spam_dataset()