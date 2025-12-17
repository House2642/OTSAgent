import pandas as pd
from embeddings import embed
from db import execute, query
from datetime import datetime, timedelta
from tqdm import tqdm

def load_data():
    df = pd.read_csv("../data/salesforce_export.csv")
    df['Schedule Date'] = pd.to_datetime(df['Schedule Date'])
    one_year_ago = datetime.now() - timedelta(days=365)
    salesforce_data = df[df['Schedule Date'] >= one_year_ago]
    print(f"✓ Loaded {len(salesforce_data)} rows (from {len(df)} total)")
    return salesforce_data

def safe_val(val):
    """Convert pandas NaN/NaT to None for PostgreSQL"""
    if pd.isna(val):
        return None
    return val

def embed_data():
    salesforce_data = load_data() 
    
    for index, row in tqdm(salesforce_data.iterrows(), total=len(salesforce_data)):
        content = f"Account: {row['Account Name']}, Opportunity: {row['Opportunity Name']}, Product: {row['Product Name']}, League: {row['League Name']}"
        embedding = embed(content)
        
        execute("""
            INSERT INTO opportunities 
            (opportunity_id, opportunity_name, opportunity_record_type, opportunity_type,
             stage, probability, product_name, product_family, league_name,
             account_name, category, agency, agency_holding_company,
             split_schedule_amount, split_expected_schedule_amount,
             created_date, schedule_date, close_date,
             loss_reason, loss_reason_context, owner_role, opportunity_owner,
             content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            safe_val(row['Opportunity ID']),
            safe_val(row['Opportunity Name']),
            safe_val(row['Opportunity Record Type']),
            safe_val(row['Opportunity Type']),
            safe_val(row['Stage']),
            safe_val(row['Probability (%)']),
            safe_val(row['Product Name']),
            safe_val(row['Product Family']),
            safe_val(row['League Name']),
            safe_val(row['Account Name']),
            safe_val(row['Category']),
            safe_val(row['Agency']),
            safe_val(row['Agency Holding Company']),
            safe_val(row['Split Schedule Amount']),
            safe_val(row['Split Expected Schedule Amount']),
            safe_val(row['Created Date']),
            safe_val(row['Schedule Date']),
            safe_val(row['Close Date']),
            safe_val(row['Loss Reason']),
            safe_val(row['Loss Reason Context']),
            safe_val(row['Owner Role']),
            safe_val(row['Opportunity Owner']),
            content,
            embedding
        ))
        
if __name__ == "__main__":
    execute("DELETE FROM opportunities")
    print("✓ Table cleared")
    embed_data()
    print("✓ Data loaded")