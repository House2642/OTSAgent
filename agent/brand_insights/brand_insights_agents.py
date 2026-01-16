from typing import TypedDict, Annotated
import operator
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition

from .demographics_agent import demographics_agent
from dotenv import load_dotenv

load_dotenv()

# agent/brand_insights/brand_insights_tools.py

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from .demographics_agent import demographics_agent


###################subagent###############
@tool
def get_demographics_insights(query: str) -> str:
    """
    Get demographic and audience insights about Overtime Sports accounts and fans.
    
    Use this tool to answer questions about:
    - Account statistics: follower counts, age breakdowns, gender splits, geography by platform
    - LDA compliance: Which accounts have 71.6%+ audience that is 21+ (required for alcohol partnerships)
    - Sports fandom: Which sports/leagues OT fans follow, watch live, play, or want to play
    - Fan overlap: Cross-league fandom (e.g., % of NBA fans who also watch WNBA)
    - Audience demographics: ethnicity, household income, employment, education, relationship status
    
    Available Overtime accounts:
    - Overtime Main, Overtime SZN, Overtime FC, Overtime Gaming
    - Overtime WBB, Overtime Select, OTE, Overtime Kicks
    - Overtime Boxing, OT7, Overtime Pulls, Cross Channel (all accounts aggregated)
    
    Examples:
    - "What's the age breakdown for OT7?"
    - "Which accounts are LDA compliant on Instagram?"
    - "How many total followers does Overtime Main have?"
    - "What percentage of NBA fans also watch WNBA?"
    - "Show me the household income distribution of OT fans"
    - "Which sports do OT fans currently play?"
    - "What's the gender split for Overtime WBB on TikTok?"
    
    Args:
        query: Natural language question about Overtime audience demographics
        
    Returns:
        Formatted insights with relevant statistics and context
    """
    result = demographics_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })
    
    # Return the last message content (the agent's final response)
    return result["messages"][-1].content

#########MAIN AGENT ########
class brand_insight_state(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

subagents = [get_demographics_insights]
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
model = llm.bind_tools(subagents)


def brand_fam_qa(state: brand_insight_state):
    response = model.invoke([SystemMessage("""
        You are a Brand Insights agent for Overtime Sports.
        
        When users ask about audience demographics, follower counts, LDA compliance,
        sports fandom, or other audience statistics, use the get_demographics_insights tool.
        
        Provide clear, concise answers with relevant data points.
        """),
        *state["messages"]])
    
    return {"messages": [response]}

execute_tools = ToolNode(subagents)
insights_graph = StateGraph(brand_insight_state)

insights_graph.add_node("QA", brand_fam_qa)
insights_graph.add_node("tools", execute_tools)

insights_graph.add_edge(START, "QA")
insights_graph.add_conditional_edges("QA", tools_condition)
insights_graph.add_edge("tools", "QA")

brand_insights_agent = insights_graph.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    print("=" * 60)
    print("BRAND INSIGHTS AGENT - Overtime Sports Audience Data")
    print("=" * 60)
    print("\nAsk me about:")
    print("  • Demographics (age, gender, ethnicity, income)")
    print("  • Follower counts by account/platform")
    print("  • LDA compliance (alcohol advertising)")
    print("  • Sports fandom and league overlap")
    print("\nType 'exit' or 'quit' to end the conversation\n")
    
    messages = []
    
    while True:
        # Get user input
        query = input("You: ").strip()
        
        # Exit conditions
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nThanks for using Brand Insights Agent!")
            break
        
        if not query:
            continue
        
        # Add user message to history
        messages.append(HumanMessage(content=query))
        
        # Invoke agent with full conversation history
        result = brand_insights_agent.invoke({
            "messages": messages
        })
        
        # Update messages with full result (includes tool calls, tool responses, etc.)
        messages = result["messages"]
        
        # Print just the final assistant response
        print(f"\nAgent: {result['messages'][-1].content}\n")

