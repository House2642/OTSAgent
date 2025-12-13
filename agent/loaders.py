import pandas as pd
from embeddings import embed
from db import execute, query
from datetime import datetime, timedelta
from tqdm import tqdm

def load_data():
    #only pull the past year
    df = pd.read_csv("../data/salesforce_export.csv")
    df['Created Date'] = pd.to_datetime(df['Created Date'])
    one_year_ago = datetime.now() - timedelta(days=365)
    salesforce_data = df[df['Created Date'] >= one_year_ago]
    return salesforce_data

def embed_data():
    salesforce_data = load_data() 
    
    for index, row in tqdm(salesforce_data.iterrows(), total=len(salesforce_data)):
        content = f"Account: {row['Account Name']}, Opportunity: {row['Opportunity Name']}, Product: {row['Product Name']}, League: {row['League Name']}"
        embedding = embed(content)
        
        execute("""
            INSERT INTO opportunities 
            (opportunity_name, opportunity_record_type, opportunity_type, 
             product_name, league_name, account_name, amount, stage, 
             probability, loss_reason, created_date, schedule_date, 
             content, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row['Opportunity Name'],
            row['Opportunity Record Type'],
            row['Opportunity Type'],
            row['Product Name'],
            row['League Name'],
            row['Account Name'],
            row['Amount'],
            row['Stage'],
            row['Probability (%)'],
            row['Loss Reason'],
            row['Created Date'],
            row['Schedule Date'],
            content,
            embedding
        ))
        
if __name__ == "__main__":
    # Clear first
    execute("DELETE FROM opportunities")
    print("✓ Table cleared")
    
    # Load all (remove limit)
    embed_data()