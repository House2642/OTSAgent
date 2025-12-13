from db import query
from embeddings import embed
from typing import Literal
from langchain_core.tools import tool

@tool
def search_opportunities(
        search_text: str, 
        stage: str = None, 
        min_amount: float = None,
        opportunity_record_type: Literal["Core", "League"] = None,
        limit: int = 10,
         ) -> list[dict]:
    """Search for opportunities semantically. Use for finding similar deals or companies."""
    embedding = embed(search_text)

    sql = """
        SELECT opportunity_name, account_name, amount, stage, 
            embedding <=> %s::vector AS distance
        FROM opportunities
        WHERE 1=1
    """
    params = [embedding]
    
    if stage:
        sql += " AND stage = %s"
        params.append(stage)
    
    if min_amount:
        sql += " AND amount >= %s"
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
        SELECT probability, opportunity_name, account_name, stage, 
                created_date, product_name, league_name
        FROM opportunities 
        WHERE account_name ILIKE %s
        ORDER BY created_date DESC;
    """
    params = (f"%{account}%",)
    retrevial = query(sql, tuple(params))

    return retrevial 

@tool
def get_account_summary(account: str) -> dict:
    """Get summary stats for an account - revenue won, deals won/lost, pipeline. Use for account overviews."""
    sql = """
        SELECT SUM(CASE WHEN stage = 'Closed Won' THEN amount ELSE 0 END) as revenue_won,
            SUM(CASE WHEN stage = 'Closed Won' THEN 1 ELSE 0 END) as deals_won,
            SUM(CASE WHEN stage = 'Closed Lost' THEN 1 ELSE 0 END) as deals_lost,
            SUM(CASE WHEN stage NOT IN ('Closed Won', 'Closed Lost') THEN 1 ELSE 0 END) as pipeline_deals,
            SUM(CASE WHEN stage NOT IN ('Closed Won', 'Closed Lost') THEN amount ELSE 0 END) as pipeline_revenue,
            SUM(CASE WHEN stage NOT IN ('Closed Won', 'Closed Lost') THEN amount*probability/100 ELSE 0 END) as weighted_pipeline_revenue
        FROM opportunities
        WHERE account_name ILIKE %s;
    """
    params = (f"%{account}%",)
    retrevial = query(sql, tuple(params))

    return retrevial[0]

if __name__ == "__main__":
    #results = search_opportunities("Nike", opportunity_record_type="Core")
    results = get_account_summary.invoke({"account": "NIKE"})
    print(results)