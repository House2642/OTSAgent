import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    
    cur.execute("DROP TABLE IF EXISTS audience_statistics")
    #cur.execute("DROP TABLE IF EXISTS opportunities")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    #opportunities table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS opportunities (
            id SERIAL PRIMARY KEY, 
            
            opportunity_id TEXT,
            opportunity_name TEXT,

            opportunity_record_type TEXT,
            opportunity_type TEXT,
            stage TEXT, 
            probability DECIMAL, 

            product_name TEXT,
            product_family TEXT, 
            league_name TEXT,

            account_name TEXT,
            category TEXT,
            agency TEXT, 
            agency_holding_company TEXT,

            created_date DATE, 
            schedule_date DATE,
            close_date DATE,
            split_schedule_amount DECIMAL, 
            split_expected_schedule_amount DECIMAL, 
            loss_reason TEXT,
            loss_reason_context TEXT, 
            owner_role TEXT, 
            opportunity_owner TEXT,
            content TEXT,
            embedding vector(1536)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_opp_record_type ON opportunities(opportunity_record_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stage ON opportunities(stage)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_schedule_date ON opportunities(schedule_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_account_name ON opportunities(account_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_product_family ON opportunities(product_family)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_probability ON opportunities(probability)")
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audience_statistics (
        id SERIAL PRIMARY KEY,
        source TEXT,
        section_id INTEGER,
        content TEXT,
        embedding vector(1536)
    )
    """)

    conn.commit()
    conn.close()
    print("✓ Database initialized")

def query(sql: str, params: tuple = None) -> list[dict]:
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(sql, params)
    results = cur.fetchall()
    conn.close()
    return [dict(r) for r in results]

def execute(sql: str, params: tuple = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Tables ready")