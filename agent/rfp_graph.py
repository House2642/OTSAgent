from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

class RFPState(TypedDict):
    raw_rfp: str
    client: str
    campaign: str
    background: str
    budget_min: str
    budget_max: str
    timing: str
    objectives: list[str]
    target_audience: str
    kpis: list[str]
    deliverables: list[str]
    other_important_notes: str
    client_history: str
    products: list[str]
    content_ideas: list[str]
    ideas_to_products: dict[str, str]
    final_proposal: str

class RFPExtract(TypedDict):
    client: str
    campaign: str
    background: str
    budget_min: str
    budget_max: str
    timing: str
    objectives: list[str]
    kpis: list[str]
    deliverables: list[str]
    target_audience: str
    other_important_notes: str

class Recommendation(TypedDict):
    products: list[str]
    content_ideas: list[str]
    ideas_to_products: dict[str, str]

def extract_rfp_info(state: RFPState) -> dict:
    extractor = llm.with_structured_output(RFPExtract)
    result = extractor.invoke([
        SystemMessage(content=f"Extract the relevant info from this Request for Proposal (RFP). Today's date is {datetime.now().strftime('%Y-%m-%d')}"),
        HumanMessage(content=state["raw_rfp"])
    ])
    return result

def gather_context(state: RFPState) -> dict:
    from tools import search_opportunities, get_account_summary
    
    client = state["client"].split("(")[0].strip()  # "Barefoot"
    
    # 1. Exact client search
    exact_opps = search_opportunities.invoke({
        "search_text": client, 
        "limit": 5
    })
    
    # Check if results actually match client
    client_opps = [o for o in exact_opps if client.lower() in o["account_name"].lower()]
    
    # 2. Category context (alcohol brands for benchmarking)
    category_opps = search_opportunities.invoke({
        "search_text": "alcohol wine beer spirits brand sponsorship",
        "limit": 10
    })
    
    # Format
    client_history = ""
    
    if client_opps:
        account_name = client_opps[0]["account_name"]
        summary = get_account_summary.invoke({"account": account_name})
        client_history += f"=== Direct Client History ({account_name}) ===\n"
        client_history += f"Revenue Won: ${summary.get('revenue_won', 0):,.0f}\n"
        for opp in client_opps:
            client_history += f"- {opp['opportunity_name']}: ${opp.get('amount', 0):,.0f} ({opp['stage']})\n"
    else:
        client_history += "=== No direct history with this client ===\n"
    
    client_history += "\n=== Similar Category Deals (Alcohol/Beverage) ===\n"
    for opp in category_opps[:5]:
        client_history += f"- {opp['account_name']}: {opp['opportunity_name']} - ${opp.get('amount', 0):,.0f} ({opp['stage']})\n"
    
    return {"client_history": client_history}

class Recommendation(TypedDict):
    products: list[str]
    content_ideas: list[str]
    ideas_to_products: dict[str, str]

def recommend_products(state: RFPState) -> dict:
    with open("../data/docs/rate_card.md", "r") as f:
        rate_card = f.read()
    
    recommender = llm.with_structured_output(Recommendation)
    
    result = recommender.invoke([
        SystemMessage(content=f"""You are an Overtime Sports sales strategist. Based on the RFP details and rate card, recommend products with creative ideas.

    RATE CARD:
    {rate_card}

    Return:
    - products: List of product names from rate card that fit their budget/objectives
    - content_ideas: Creative concepts specific to their brand/campaign
    - ideas_to_products: Map each content idea to the product it uses

    Stay within budget. Prioritize content-first."""),
            
    HumanMessage(content=f"""
    CLIENT: {state['client']}
    CAMPAIGN: {state['campaign']}
    BACKGROUND: {state['background']}
    BUDGET: {state['budget_min']} - {state['budget_max']}
    TIMING: {state['timing']}
    OBJECTIVES: {state['objectives']}
    TARGET AUDIENCE: {state['target_audience']}
    CLIENT HISTORY: {state['client_history']}
    """)
        ])
        
    return result

# Build graph
rfp_graph = StateGraph(RFPState)
rfp_graph.add_node("extract", extract_rfp_info)
rfp_graph.add_node("pastContext", gather_context)
rfp_graph.add_node("recommend", recommend_products)
rfp_graph.add_edge(START, "extract")
rfp_graph.add_edge("extract", "pastContext")
rfp_graph.add_edge("pastContext","recommend") 
rfp_graph.add_edge("recommend", END) 

rfp_app = rfp_graph.compile()

if __name__ == "__main__":
    test_rfp = open("../data/docs/rfp.md").read()
    result = rfp_app.invoke({"raw_rfp": test_rfp})
    
    for key, value in result.items():
        if key != "raw_rfp":
            print(f"{key}: {value}")