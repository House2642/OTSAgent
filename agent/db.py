import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def create_salesforce_opportunities(cur_conn):
    cur_conn.execute("DROP TABLE IF EXISTS opportunities")
    cur_conn.execute("""
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
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_opp_record_type ON opportunities(opportunity_record_type)")
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_stage ON opportunities(stage)")
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_schedule_date ON opportunities(schedule_date)")
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_account_name ON opportunities(account_name)")
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_product_family ON opportunities(product_family)")
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_probability ON opportunities(probability)")

def create_channel_stats_table(cur_conn):
    """
    Create platform demographics table matching CSV structure.
    Each row = one demographic stat across all platforms.
    
    Source: demographics_final.csv
    """
    cur_conn.execute("DROP TABLE IF EXISTS channel_stats")
    
    cur_conn.execute("""
        CREATE TABLE IF NOT EXISTS channel_stats (
            id SERIAL PRIMARY KEY,
            youtube_no_shorts FLOAT,
            all_youtube FLOAT,
            instagram FLOAT,
            facebook FLOAT,
            twitter FLOAT,
            snapchat FLOAT,
            tiktok FLOAT,
            stat_type TEXT NOT NULL,         -- Age, Gender, Geography, etc.
            stat_value TEXT NOT NULL,        -- 18-24, Male, US, etc.
            account TEXT NOT NULL            -- Overtime Main
        )
    """)
    
    # Indexes for filtering
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_stat_type ON channel_stats(stat_type)")
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_account ON channel_stats(account)")

def create_sports_fandom_table(cur_conn):
    """
    Create sports fandom table matching CSV structure.
    Source: sports_fandom.csv
    Columns: Sport/League Name, Metric Type, OT Fans, OT Index, Category, Source
    """
    cur_conn.execute("DROP TABLE IF EXISTS sports_fandom")
    
    cur_conn.execute("""
        CREATE TABLE IF NOT EXISTS sports_fandom (
            id SERIAL PRIMARY KEY,
            sport_league_name TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            ot_fans FLOAT,
            ot_index INTEGER,
            category TEXT,
            source TEXT
        )
    """)
    
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_sport_league ON sports_fandom(sport_league_name)")
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_type ON sports_fandom(metric_type)")


def create_sports_overlap_table(cur_conn):
    """
    Create sports overlap table matching CSV structure.
    Source: sports_fandom_overlap.csv
    Columns: League_Primary, League_Secondary, Overlap_Percentage, Source
    """
    cur_conn.execute("DROP TABLE IF EXISTS sports_overlap")
    
    cur_conn.execute("""
        CREATE TABLE IF NOT EXISTS sports_overlap (
            id SERIAL PRIMARY KEY,
            league_primary TEXT NOT NULL,
            league_secondary TEXT NOT NULL,
            overlap_percentage FLOAT,
            source TEXT
        )
    """)
    
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_league_primary ON sports_overlap(league_primary)")
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_league_secondary ON sports_overlap(league_secondary)")


def create_ot_fan_demographics_table(cur_conn):
    """
    Create OT fan demographics table matching CSV structure.
    Source: ot_fan_demos.csv
    Columns: OT Fans, Gen Pop Average, OT Index, Data Category, Data Value, Source
    """
    cur_conn.execute("DROP TABLE IF EXISTS ot_fan_demographics")
    
    cur_conn.execute("""
        CREATE TABLE IF NOT EXISTS ot_fan_demographics (
            id SERIAL PRIMARY KEY,
            ot_fans FLOAT,
            gen_pop_average FLOAT,
            ot_index FLOAT,
            data_category TEXT NOT NULL,
            data_value TEXT NOT NULL,
            source TEXT
        )
    """)
    
    cur_conn.execute("CREATE INDEX IF NOT EXISTS idx_data_category ON ot_fan_demographics(data_category)")

def init_db(debug: bool):
    conn = get_conn()
    cur = conn.cursor()
    
    #opportunities table
    create_salesforce_opportunities(cur)
    if debug:
        print("created sales force table")

    create_channel_stats_table(cur)
    if debug:
        print("created channel demograpics table")
    
    create_ot_fan_demographics_table(cur)
    if debug:
        print("created demographics table")
    
    create_sports_fandom_table(cur)
    if debug:
        print("created fandom table")

    create_sports_overlap_table(cur)
    if debug:
        print("created fandom overalp table")

    conn.commit()
    conn.close()
    if debug:
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
    init_db(debug=True)
    print("Tables ready")