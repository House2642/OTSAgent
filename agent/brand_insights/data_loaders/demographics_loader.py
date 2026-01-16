
import sys
import os

#add agent/ directory to Python path
current_file = os.path.abspath(__file__)
data_loaders_dir = os.path.dirname(current_file)
brand_insights_dir = os.path.dirname(data_loaders_dir)
agent_dir = os.path.dirname(brand_insights_dir)
sys.path.insert(0,agent_dir)

DEBUG = True

import pandas as pd
from db import execute
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
DATADIR = os.getenv("DATADIR")

def safe_val(val):
    if pd.isna(val):
        return None
    return val

def load_channel_stats():
    channel_stats = pd.read_csv(f"{DATADIR}channel_stats.csv")  # Use correct filename
    
    for index, row in tqdm(channel_stats.iterrows(), total=len(channel_stats)):  # Missing closing paren
        execute("""
            INSERT INTO channel_stats(
                youtube_no_shorts, all_youtube, 
                instagram, facebook, twitter, snapchat, tiktok,
                stat_type, stat_value, account
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            safe_val(row['Youtube (No Shorts)']),      # Use 'row' not 'df'
            safe_val(row['All YouTube (via Tubular, Shorts, inc)']),
            safe_val(row['Instagram']),
            safe_val(row['Facebook']),
            safe_val(row['Twitter']),
            safe_val(row['Snapchat']),
            safe_val(row['TikTok']),
            safe_val(row['Stat Type']),
            safe_val(row['Stat Value']),
            safe_val(row['Account'])
        ))
    if DEBUG:
        print(f"Loaded {len(channel_stats)} platform demographic records")

def load_sports_fandom():
    sports_fandom = pd.read_csv(f"{DATADIR}sports_fandom.csv")
    
    for index, row in tqdm(sports_fandom.iterrows(), total=len(sports_fandom)):
        execute("""
            INSERT INTO sports_fandom(
                sport_league_name, metric_type, ot_fans, 
                ot_index, category, source
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            safe_val(row['Sport/League Name']),
            safe_val(row['Metric Type']),
            safe_val(row['OT Fans']),
            safe_val(row['OT Index']),
            safe_val(row['Category']),
            safe_val(row['Source'])
        ))
    
    if DEBUG:
        print(f"Loaded {len(sports_fandom)} sports fandom records")


def load_sports_overlap():
    sports_overlap = pd.read_csv(f"{DATADIR}sports_fandom_overlap.csv")
    
    for index, row in tqdm(sports_overlap.iterrows(), total=len(sports_overlap)):
        execute("""
            INSERT INTO sports_overlap(
                league_primary, league_secondary, 
                overlap_percentage, source
            )
            VALUES (%s, %s, %s, %s)
        """,
        (
            safe_val(row['League_Primary']),
            safe_val(row['League_Secondary']),
            safe_val(row['Overlap_Percentage']),
            safe_val(row['Source'])
        ))
    
    if DEBUG:
        print(f"Loaded {len(sports_overlap)} sports overlap records")


def load_ot_fan_demographics():
    ot_fan_demos = pd.read_csv(f"{DATADIR}ot_fan_demos.csv")
    
    for index, row in tqdm(ot_fan_demos.iterrows(), total=len(ot_fan_demos)):
        execute("""
            INSERT INTO ot_fan_demographics(
                ot_fans, gen_pop_average, ot_index,
                data_category, data_value, source
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            safe_val(row['OT Fans']),
            safe_val(row['Gen Pop Average']),
            safe_val(row['OT Index']),
            safe_val(row['Data Category']),
            safe_val(row['Data Value']),
            safe_val(row['Source'])
        ))
    
    if DEBUG:
        print(f"Loaded {len(ot_fan_demos)} OT fan demographic records")


# In your __main__ section:
if __name__ == "__main__":
    DEBUG = True
    
    load_channel_stats()
    load_sports_fandom()
    load_sports_overlap()
    load_ot_fan_demographics()
    if DEBUG:
        print("All demographics data loaded!")

