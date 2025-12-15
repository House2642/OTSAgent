from typing import TypedDict, Annotated
import operator
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import SystemMessage, HumanMessage
from tools import search_opportunities, get_account_history, get_account_summary
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
load_dotenv()

salesforce_tools = [search_opportunities, get_account_history, get_account_summary]
model = ChatAnthropic(model="claude-haiku-4-5-20251001")
salesforce_llm = model.bind_tools(salesforce_tools)
#Nodes

class RFPState(TypedDict):
    messages: Annotated[str, add_messages]

def retrieve_salesforce(state: RFPState):
    get_data = salesforce_llm.invoke([SystemMessage(content="""
            You are an RFP assistant for Overtime Sports. You help answer questions about past deals and accounts.

            Use your tools to get accurate data:
            - search_opportunities: Find similar deals or companies semantically
            - get_account_history: Get all deals for a specific company
            - get_account_summary: Get summary stats (revenue won, deals won/lost, pipeline)

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
        result = app.invoke({"messages": messages})
        
        # Update messages with full conversation
        messages = result["messages"]
        
        # Print agent response
        print(f"\nAgent: {messages[-1].content}")