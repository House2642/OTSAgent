
import sys
import os

#add agent/ directory to Python path
current_file = os.path.abspath(__file__)
brand_insights_dir = os.path.dirname(current_file)
agent_dir = os.path.dirname(brand_insights_dir)
sys.path.insert(0,agent_dir)

from db import query
from langchain_core.tools import tool

@tool
def account_search(account: str, stat_type: str) -> list[dict]:
    """
    Get audience statistics for a specific Overtime Sports account/channel.
    
    Use this when users ask about demographics or follower counts for 
    specific Overtime properties (OTE, OT7, Overtime Main, etc.) or 
    when comparing stats across all channels.
    
    Args:
        account: The Overtime account/property. Options:
                 'Overtime Main', 'Overtime SZN', 'Overtime FC', 
                 'Overtime Gaming', 'Overtime WBB', 'Overtime Select', 
                 'OTE', 'Overtime Kicks', 'Overtime Boxing', 'OT7', 
                 'Overtime Pulls', 'Cross Channel'
                 Note: 'Cross Channel' aggregates data across all accounts.
        stat_type: The demographic statistic to retrieve. Options:
                   'Total Followers', 'Age', 'Combined_Age', 
                   'Gender', 'Geography'
    
    Examples:
    - "What's the age breakdown for OTE?"
    - "How many followers does Overtime Main have?"
    - "Show me gender demographics for OT7"
    - "What's the overall age distribution across all channels?"
      → Use account='Cross Channel'
    - "Compare male vs female audience for Overtime WBB"
    -"Which accounts have more that 50% 21+"
      → If you want to get combined ages like 21+ use the stat_type 

    
    Returns list of demographic records with stat values and percentages.
    """
    results = query("""
        SELECT * FROM channel_stats
        WHERE account = %s AND stat_type = %s
    """, params=(account, stat_type))
    
    return results

