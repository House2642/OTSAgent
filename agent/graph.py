from typing import TypedDict, Annotated
import operator
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from tools import search_opportunities, get_account_history, get_account_summary, get_revenue, get_pipeline_by_stage, get_pipeline, get_deals
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

salesforce_tools = [search_opportunities, get_account_history, get_account_summary, get_revenue, get_pipeline_by_stage, get_pipeline, get_deals]
model = ChatAnthropic(model="claude-haiku-4-5-20251001")
salesforce_llm = model.bind_tools(salesforce_tools)
#Nodes

class RFPState(TypedDict):
    messages: Annotated[str, add_messages]

def retrieve_salesforce(state: RFPState):
    get_data = salesforce_llm.invoke([SystemMessage(content=f"""
        You are an RFP assistant for Overtime Sports. You help answer questions about past deals, accounts, and pipeline.
        Current Date is {datetime.now().strftime("%Y-%m-%d")}
        TOOLS:
        - search_opportunities: Find similar deals or companies semantically
        - get_account_history: Get all deals for a specific company
        - get_account_summary: Get summary stats for an account (revenue won, pipeline)
        - get_revenue: Get booked revenue with filters (date range, product family, account, Core/League)
        - get_pipeline: Get weighted/unweighted pipeline totals (can filter by min probability for "booked 75" questions)
        - get_pipeline_by_stage: Get pipeline breakdown by stage
        - get_deals: Get a complete list of ALL deals matching filters (use for "show me all Q3 deals")

        KEY CONCEPTS:
        - "Booked" means Closed Won (Core) or Long-Form Contract Signed (League)
        - Always use schedule_date for revenue timing - revenue is recognized when scheduled, not when deal closed
        - Core and League are separate business lines - filter by opportunity_record_type when asked about one specifically
        - Dates should be YYYY-MM-DD format

        COMMON QUESTION PATTERNS:

        "How much has [account] booked this year?"
        → Use get_revenue with account and date range

        "What's our booked 50/75/90 this quarter?"
        → This means booked revenue PLUS weighted pipeline at that probability or above
        → Call get_revenue for booked amount, then get_pipeline with min_probability for pipeline portion
        → Also call get_pipeline_by_stage to show breakdown
        → Provide total first, then breakdown by stage

        "How much media did we book?"
        → Use get_revenue with product_family="Media"

        "How much for Overtime Elite / OTE?"
        → Use get_revenue with product_family="OTE League Offering"
        → League product families:
        - OTE = "OTE League Offering"
        - OT7 = "OT7 League Offering"  
        - OT Select = "OTS League Offering"
        - OTX = "OTX League Offering"
        - OT Nationals = "OT Nationals League Offering"
        → Don't use league_name field - use product_family instead (handles multi-league deals)

        "What's our pipeline for Core/League?"
        → Use get_pipeline or get_pipeline_by_stage with opportunity_record_type

        Always use tools for data - never make up numbers.
        """), *state["messages"]])
    
    return {"messages": [get_data]}

execute_tools = ToolNode(salesforce_tools)

salesbot = StateGraph(RFPState)
salesbot.add_node("QA", retrieve_salesforce)
salesbot.add_node("tools", execute_tools)

salesbot.add_edge(START, "QA")
salesbot.add_conditional_edges("QA", tools_condition)
salesbot.add_edge("tools", "QA")

sf_app = salesbot.compile()

if __name__ == "__main__":
    messages = []
    
    print("Salesforce Agent (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        messages.append(HumanMessage(content=user_input))
        result = sf_app.invoke({"messages": messages})
        
        # Update messages with full conversation
        messages = result["messages"]
        
        # Print agent response
        print(f"\nAgent: {messages[-1].content}")