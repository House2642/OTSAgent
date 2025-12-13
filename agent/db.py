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
    #cur.execute("DROP TABLE IF EXISTS opportunities")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    #opportunities table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS opportunities (
            id SERIAL PRIMARY KEY, 
            opportunity_name TEXT, 
            opportunity_record_type TEXT,
            opportunity_type TEXT,
            product_name TEXT,
            league_name TEXT,
            account_name TEXT,
            amount DECIMAL,
            stage TEXT,
            probability DECIMAL,
            loss_reason TEXT,
            created_date DATE,
            schedule_date DATE,
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