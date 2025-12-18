from db import query
from embeddings import embed
from typing import Literal
from langchain_core.tools import tool
StageType = Literal[
    # Core stages
    "Closed Lost",
    "Proactive Pitch",
    "RFP Activated", 
    "Proposal Q&A",
    "Specs Requested",
    "Verbal",
    "Closed Won",
    # League stages
    "Proactive Outreach",
    "Proposal Delivered",
    "Proposal Feedback Received",
    "Proposal Accepted",
    "Long-Form Contract In-Progress",
    "Long-Form Contract Signed",
]

RecordType = Literal["Core", "League"]

@tool
def search_opportunities(
        search_text: str, 
        stage: StageType = None, 
        min_amount: float = None,
        opportunity_record_type: RecordType = None,
        limit: int = 10,
         ) -> list[dict]:
    """Search for opportunities semantically. Use for finding similar deals or companies."""
    embedding = embed(search_text)

    sql = """
        SELECT opportunity_name, account_name, split_schedule_amount, stage,
            product_family, schedule_date, 
            embedding <=> %s::vector AS distance
        FROM opportunities
        WHERE 1=1
    """
    params = [embedding]
    
    if stage:
        sql += " AND stage = %s"
        params.append(stage)
    
    if min_amount:
        sql += " AND split_schedule_amount >= %s"
        params.append(min_amount)
    
    if opportunity_record_type:
        sql += " AND opportunity_record_type = %s"
        params.append(opportunity_record_type)
    
    sql += " ORDER BY distance LIMIT %s"
    params.append(limit)

    retrieval = query(sql, tuple(params))
    
    return retrieval

@tool
def get_account_history(account: str) -> list[dict]:
    """Get all deals for a specific account. Use when asked about a specific company's history."""
    sql = """
        SELECT probability, opportunity_name, stage, account_name,
                split_schedule_amount, schedule_date, product_name,
                product_family, league_name
        FROM opportunities 
        WHERE account_name ILIKE %s
        ORDER BY schedule_date DESC;
    """
    
    return query(sql, (f"%{account}%",)) 

@tool
def get_account_summary(account: str) -> dict:
    """Get summary stats for an account - revenue won, deals won/lost, pipeline. Use for account overviews."""
    sql = """
        SELECT SUM(CASE WHEN stage IN ('Closed Won', 'Long-Form Contract Signed') THEN split_schedule_amount ELSE 0 END) as revenue_won,
            SUM(CASE WHEN stage IN ('Closed Won', 'Long-Form Contract Signed') THEN 1 ELSE 0 END) as deals_won,
            SUM(CASE WHEN stage = 'Closed Lost' THEN 1 ELSE 0 END) as deals_lost,
            SUM(CASE WHEN stage NOT IN ('Closed Won', 'Closed Lost', 'Long-Form Contract Signed') THEN 1 ELSE 0 END) as pipeline_deals,
            SUM(CASE WHEN stage NOT IN ('Closed Won', 'Closed Lost', 'Long-Form Contract Signed') THEN split_schedule_amount ELSE 0 END) as pipeline_revenue,
            SUM(CASE WHEN stage NOT IN ('Closed Won', 'Closed Lost', 'Long-Form Contract Signed') THEN split_expected_schedule_amount ELSE 0 END) as weighted_pipeline_revenue
        FROM opportunities
        WHERE account_name ILIKE %s or opportunity_name ILIKE %s;
    """
    results = query(sql, (f"%{account}%",f"%{account}%"))
    return results[0]

@tool 
def get_revenue(
    opportunity_record_type: RecordType = None,
    product_family: str = None,
    account: str = None,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """
    Get booked revenue (Closed Won / Long-Form Contract Signed deals only).
    Use for questions like "how much did we book in Q3?" or "how much media revenue this year?"
    Dates should be YYYY-MM-DD format.
    """
    sql = """
        SELECT 
            SUM(split_schedule_amount) as booked_revenue,
            COUNT(DISTINCT opportunity_id) as deal_count
        FROM opportunities
        WHERE stage IN ('Closed Won', 'Long-Form Contract Signed')
    """
    params = []
    
    if opportunity_record_type:
        sql += " AND opportunity_record_type = %s"
        params.append(opportunity_record_type)
    
    if product_family:
        sql += " AND product_family ILIKE %s"
        params.append(f"%{product_family}%")
    
    if account:
        sql += " AND (account_name ILIKE %s OR opportunity_name ILIKE %s)"
        params.append(f"%{account}%")
        params.append(f"%{account}%")
    
    if start_date:
        sql += " AND schedule_date >= %s"
        params.append(start_date)
    
    if end_date:
        sql += " AND schedule_date <= %s"
        params.append(end_date)
    
    return query(sql, tuple(params))[0]

@tool
def get_pipeline(
    opportunity_record_type: RecordType = None,
    min_probability: int = None,
    account: str = None,
    start_date: str = None,
    end_date: str = None,
) -> dict:
    """
    Get pipeline stats - weighted and unweighted revenue for open deals.
    Use for questions like "what's our booked 75?" or "what's our weighted pipeline for Core?"
    min_probability filters to deals at that probability or above (e.g., 75 for "booked 75").
    Dates should be YYYY-MM-DD format.
    """
    sql = """
        SELECT 
            SUM(split_schedule_amount) as unweighted_revenue,
            SUM(split_expected_schedule_amount) as weighted_revenue,
            COUNT(DISTINCT opportunity_id) as deal_count
        FROM opportunities
        WHERE stage NOT IN ('Closed Won', 'Long-Form Contract Signed', 'Closed Lost')
    """
    params = []
    
    if opportunity_record_type:
        sql += " AND opportunity_record_type = %s"
        params.append(opportunity_record_type)
    
    if min_probability:
        sql += " AND probability >= %s"
        params.append(min_probability)
    
    if account:
        sql += " AND (account_name ILIKE %s OR opportunity_name ILIKE %s)"
        params.append(f"%{account}%")
        params.append(f"%{account}%")
    
    if start_date:
        sql += " AND schedule_date >= %s"
        params.append(start_date)
    
    if end_date:
        sql += " AND schedule_date <= %s"
        params.append(end_date)
    
    return query(sql, tuple(params))[0]

@tool
def get_deals(
    opportunity_record_type: RecordType = None,
    stage: StageType = None,
    product_family: str = None,
    account: str = None,
    start_date: str = None,
    end_date: str = None,
    limit: int = 50,
) -> list[dict]:
    """
    Get a comprehensive list of ALL deals matching filters. Use when you need every deal
    for a time period, not just semantically similar ones. Returns unique deals with totals.
    Dates should be YYYY-MM-DD format.
    """
    sql = """
        SELECT 
            opportunity_name, account_name, stage, product_family,
            SUM(split_schedule_amount) as total_amount
        FROM opportunities
        WHERE 1=1
    """
    params = []
    
    if opportunity_record_type:
        sql += " AND opportunity_record_type = %s"
        params.append(opportunity_record_type)
    
    if stage:
        sql += " AND stage = %s"
        params.append(stage)
    
    if product_family:
        sql += " AND product_family ILIKE %s"
        params.append(f"%{product_family}%")
    
    if account:
        sql += " AND (account_name ILIKE %s OR opportunity_name ILIKE %s)"
        params.append(f"%{account}%")
        params.append(f"%{account}%")
    
    if start_date:
        sql += " AND schedule_date >= %s"
        params.append(start_date)
    
    if end_date:
        sql += " AND schedule_date <= %s"
        params.append(end_date)
    
    sql += """
        GROUP BY opportunity_name, account_name, stage, product_family
        ORDER BY total_amount DESC
        LIMIT %s
    """
    params.append(limit)
    
    return query(sql, tuple(params))

@tool
def get_pipeline_by_stage(
    opportunity_record_type: RecordType = None,
    account: str = None,
    start_date: str = None,
    end_date: str = None,
) -> list[dict]:
    """
    Get pipeline breakdown by stage. Shows weighted and unweighted amounts at each stage.
    Use when asked for pipeline breakdown or "how much at each stage?"
    Dates should be YYYY-MM-DD format.
    """
    sql = """
        SELECT 
            stage,
            probability,
            SUM(split_schedule_amount) as unweighted_revenue,
            SUM(split_expected_schedule_amount) as weighted_revenue,
            COUNT(DISTINCT opportunity_id) as deal_count
        FROM opportunities
        WHERE stage NOT IN ('Closed Won', 'Long-Form Contract Signed', 'Closed Lost')
    """
    params = []
    
    if opportunity_record_type:
        sql += " AND opportunity_record_type = %s"
        params.append(opportunity_record_type)
    
    if account:
        sql += " AND (account_name ILIKE %s OR opportunity_name ILIKE %s)"
        params.append(f"%{account}%")
        params.append(f"%{account}%")
    
    if start_date:
        sql += " AND schedule_date >= %s"
        params.append(start_date)
    
    if end_date:
        sql += " AND schedule_date <= %s"
        params.append(end_date)
    
    sql += " GROUP BY stage, probability ORDER BY probability DESC"
    
    return query(sql, tuple(params))

@tool
def search_audience_data(search_query: str, limit: int = 2) -> list[dict]:
    """Search audience research docs for demographics, behaviors, preferences."""
    embedding = embed(search_query)
    sql = """
        SELECT source, content, embedding <=> %s::vector AS distance
        FROM audience_statistics
        ORDER BY distance
        LIMIT %s
    """
    return query(sql, (embedding, limit))

if __name__ == "__main__":
    #results = search_opportunities("Nike", opportunity_record_type="Core")
    results = get_account_summary.invoke({"account": "NIKE"})
    print(results)