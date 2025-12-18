from typing import TypedDict
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv
from datetime import datetime
from tools import search_audience_data

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
    audience_stats: list[str]
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

class audienceStats(TypedDict):
    stats: list[str]

def extract_rfp_info(state: RFPState) -> dict:
    extractor = llm.with_structured_output(RFPExtract)
    result = extractor.invoke([
        SystemMessage(content=f"Extract the relevant info from this Request for Proposal (RFP). Today's date is {datetime.now().strftime('%Y-%m-%d')}"),
        HumanMessage(content=state["raw_rfp"])
    ])
    return result

def gather_sales_context(state: RFPState) -> dict:
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

def get_audience_statistics(state: RFPState) -> dict:
    messages = [
        SystemMessage(content=f"""You are an Overtime Sports Sales strategist building a pitch for {state['client']}.

            You MUST call search_audience_data at least 4 times with DIFFERENT queries:

            REQUIRED SEARCHES:
            1. Demographics: "age demographics A18-24 A21-34 A18-34 gender male female audience reach"
            2. Category behavior
            3. Purchase intent: "brand partnerships advertising purchase intent influence likely to purchase"
            4. Social/lifestyle: "social media shopping influence inspiration athletes influencers"

            RETURN EXACTLY 6 STATS - each must be UNIQUE (no duplicates with different wording):

            1. TARGET DEMO REACH (required: age INDEX)
            Format: "[X] INDEX on [age demo] ([Y]% of OT audience)"
            Example: "488 INDEX on A18-24 (45% of OT audience)"
            
            2. TARGET DEMO REACH (required: second demo - gender, ethnicity, or lifestyle)
            Format: "[X] INDEX on [segment]" or "[X]MM [segment]"
            Example: "239 INDEX Multicultural audience" or "25MM F21-54"

            3. CATEGORY BEHAVIOR - How OT fans engage with {state['client']}'s category
            Format: "OT fans are [X]x more likely to [specific behavior relevant to {state['client']}]"
            Example: "OT fans are 3x more likely to purchase alcohol at sporting events"

            4. PURCHASE INTENT - Brand consideration after seeing OT partnership
            Format: "[X]% of OT fans are more likely to purchase from [category] brands after seeing them with Overtime ([Y] INDEX)"
            Example: "66% of OT fans are more likely to purchase an alcoholic beverage when they see it advertising with Overtime (346 INDEX)"

            5. CATEGORY AFFINITY - Over-index on category-specific behavior  
            Format: "[X] INDEX - [behavior that maps to {state['client']}'s product]"
            Example: "557 INDEX - shop via social media for inspiration from athletes"

            6. BRAND IMPACT - OT partnership effectiveness
            Format: "[X]% of OT fans [action] when they see a brand partner with Overtime"
            Example: "78% of OT fans are more interested in a brand when they see it working with Overtime"

            CRITICAL RULES:
            - Stats 1-2 MUST have demographic INDEX numbers (age, gender, ethnicity)
            - Do NOT repeat the same stat with different wording
            - Do NOT include generic "OT fans are sports fans" stats
            - Do NOT include stats unrelated to {state['client']}'s category
            - Every stat needs a number (INDEX, %, or reach)
            """),
                HumanMessage(content=f"Target audience:\n{state['target_audience']}\n\nCampaign objectives:\n{state.get('objectives', '')}")
            ]

    # 1) Let the model decide what to search (tool calling enabled)
    tool_llm = llm.bind_tools([search_audience_data])
    first = tool_llm.invoke(messages)

    # 2) Execute any tool calls and add ToolMessage(s)
    tool_messages = []
    for tc in (getattr(first, "tool_calls", None) or []):
        print(tc)
        tool_result = search_audience_data.invoke(tc["args"])
        print(tool_result)
        tool_messages.append(
            ToolMessage(
                tool_call_id=tc["id"],
                content=str(tool_result),
            )
        )

    # 3) Final pass: force structured output using the tool results
    final_llm = llm.with_structured_output(audienceStats)
    final = final_llm.invoke(messages + [first] + tool_messages)

    # final is like: {"stats": [...]}
    return {"audience_stats": final["stats"]}

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
rfp_graph.add_node("pastContext", gather_sales_context)
rfp_graph.add_node("recommend", recommend_products)
rfp_graph.add_node("audienceStats", get_audience_statistics)

rfp_graph.add_edge(START, "extract")
rfp_graph.add_edge("extract", "pastContext")
rfp_graph.add_edge("pastContext","audienceStats")
rfp_graph.add_edge("audienceStats","recommend") 
rfp_graph.add_edge("recommend", END) 

rfp_app = rfp_graph.compile()

if __name__ == "__main__":
    test_rfp = open("../data/docs/rfp.md").read()
    result = rfp_app.invoke({"raw_rfp": test_rfp})
    
    for key, value in result.items():
        if key != "raw_rfp":
            print(f"{key}: {value}")