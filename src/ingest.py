import os
import time
import pandas as pd
from chembl_webresource_client.new_client import new_client

def fetch_target_data(target_id: str, filename: str):
    # Start the clock
    start_time = time.time()
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_path, "data", "raw")
    full_file_path = os.path.join(raw_data_path, filename)
    os.makedirs(raw_data_path, exist_ok=True)

    print(f"🚀 Initializing Ingestion for Target: {target_id}...")

    try:
        activity = new_client.activity
        # Filtering for IC50 bioactivity data
        res = activity.filter(target_chembl_id=target_id).filter(standard_type="IC50")
        
        # Ingestion happens here
        df = pd.DataFrame.from_dict(res)
        
        if not df.empty:
            df.to_csv(full_file_path, index=False)
            
            # End the clock
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ SUCCESS!")
            print(f"📊 Rows Ingested: {len(df)}")
            print(f"⏱️  Time Elapsed: {duration:.2f} seconds")
            print(f"📁 Path: {full_file_path}")
        else:
            print("⚠️ FAILED: No data found for this target.")

    except Exception as e:
        print(f"❌ AN ERROR OCCURRED: {e}")

if __name__ == "__main__":
    # Let's try MMP9 again
    fetch_target_data("CHEMBL392", "mmp9.csv")