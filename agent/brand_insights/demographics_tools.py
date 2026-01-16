
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

@tool
def fan_overlap(league_1: str, league_2:str) -> list[dict]:
    """
    The format is % of league_1 overtime fans are also fans of league_2
    leagues: NBA, NFL, NCAAF, NCAAB, MLB, NHL, WNBA

    Examples:
    How many NBA overtime fans are also WNBA fans?
    """
    results = query("""
        SELECT * FROM sports_overlap
        WHERE league_primary = %s and league_secondary = %s
    """, params=(league_1, league_2))
    
    return results

@tool
def sports_fandom(sport_or_league:str, metric_type:str) -> list[dict]:
    """
    Use this tool to understand which sports and leagues the overtime audience are avid fans of, they watch live and stream live, and which
    sports they currently play + sports they want to play
    sports_or_league: Football, Basketball, MMA/UFC, Boxing, Baseball, Soccer, 
                  Gaming (e-sports), Wrestling, Motorsports, Ice Hockey, Golf, Volleyball, Tennis, Softball, 
                Lacrosse, Pickleball, Rugby, Cricket, NFL, NBA, MLB, College Football, Men's College Basketball, FIFA World Cup, UEFA Champions League, 
                MLS, NHL, European Soccer, Formula 1, HS Football, Copa America, Boys HS Basketball, Premier Lacrosse League, WNBA, Women's College Basketball, 
                FIFA Women's World Cup, NWSL, UEFA Women's Champions League, European Women's Soccer, Professional Women's Hockey, Girls HS Basketball
    metric_type: 'Avid Fandom', 'Watch Live', 'Stream Live', 'Currently Play', 'Currently + Want to Play'
    to qualify for significant Over-Index the OT index must be 150+
    """

    results = query("""
        SELECT * FROM sports_fandom
        WHERE sport_league_name = %s AND metric_type = %s
    """, params=(sport_or_league, metric_type))

    return results

@tool
def audience_demographics(data_category:str) -> list[dict]:
    """
    Use this tool to understand the ethinicity, household income, employment, living situation, education, language, and relationshipstatus/children 
    of the overtime sports audience

    data_category: 'Ethnicity', 'Household Income', 'Employment', 'Home Ownership/Living Situation', 'Relationship Status/Children', 'Languages', 'Students/Highest Degree'
    """
    
    results = query("""
        SELECT * FROM ot_fan_demographics
        WHERE data_category = %s
    """, params=(data_category))
    
    return results