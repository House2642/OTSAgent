import pandas as pd
from embeddings import embed
from db import execute, query
from datetime import datetime, timedelta
from tqdm import tqdm
import os
from docx import Document
import json
from pathlib import Path

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

def load_audience_statistics(folder_path:str):
    for filename in os.listdir(folder_path):
        if not filename.endswith(".docx"):
            continue
        
        doc = Document(f"{folder_path}/{filename}")
        content = "\n".join([p.text for p in doc.paragraphs])
        
        # Chunk by sections (split on headers/bullets)
        chunks = chunk_by_section(content, max_chars=1500)
        
        for i, chunk in enumerate(chunks):
            embedding = embed(chunk)
            execute("""
                INSERT INTO audience_statistics (source, section_id, content, embedding)
                VALUES (%s, %s, %s, %s)
            """, (filename, i, chunk, embedding))
        print(f"✓ {filename}")

def chunk_by_section(content: str, max_chars: int = 1500) -> list[str]:
    """Split on double newlines, merge small chunks, split large ones"""
    raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    
    chunks = []
    current = ""
    
    for chunk in raw_chunks:
        if len(current) + len(chunk) < max_chars:
            current += "\n\n" + chunk if current else chunk
        else:
            if current:
                chunks.append(current)
            # If single chunk is too big, just truncate it
            current = chunk[:max_chars] if len(chunk) > max_chars else chunk
    
    if current:
        chunks.append(current)
    
    return [c for c in chunks if len(c) > 100]

def load_power_stats(filepath: str):
    """Load the Power Stats xlsx - each row is already a chunk"""
    df = pd.read_excel(filepath, skiprows=1)  # Skip the "Last updated" row
    df.columns = ['Topic', 'Power_Stat', 'Source', 'Year']
    df = df.dropna(subset=['Power_Stat'])  # Remove empty rows
    
    for _, row in df.iterrows():
        content = f"Topic: {row['Topic']}\nStat: {row['Power_Stat']}\nSource: {row['Source']}"
        embedding = embed(content)
        
        execute("""
            INSERT INTO audience_statistics (source, section_id, content, embedding)
            VALUES (%s, %s, %s, %s)
        """, ('Power_Stats.xlsx', 0, content, embedding))
        
    print(f"✓ Loaded {len(df)} power stats")

def load_unstructured_json(json_path: str):
    """Load a single unstructured JSON file into the database"""
    with open(json_path) as f:
        elements = json.load(f)
    
    filename = Path(json_path).stem
    loaded = 0
    
    for el in tqdm(elements, desc=filename[:30]):
        if el.get('type') != 'Table':
            continue
            
        text = el.get('text', '').strip()
        if len(text) < 30:
            continue
            
        page_name = el.get('metadata', {}).get('page_name', 'Unknown')
        
        content = f"[{page_name}] {text}"
        embedding = embed(content)
        
        execute("""
            INSERT INTO audience_statistics (source, section_id, content, embedding)
            VALUES (%s, %s, %s, %s)
            """, (filename, 0, content, embedding))
        
        loaded += 1
    
    print(f"✓ {filename}: {loaded} tables loaded")
    return loaded

if __name__ == "__main__":
    #execute("DELETE FROM opportunities")
    #print("✓ Table cleared")
    #embed_data()
    #print("✓ Data loaded")
    execute("DELETE FROM audience_statistics")
    for filename in os.listdir("../data/docs/unstructured_docs/"):
        load_unstructured_json(f"../data/docs/unstructured_docs/{filename}")
    print(query("SELECT source FROM audience_statistics"))