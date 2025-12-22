from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv
from datetime import datetime
from tools import get_relevant_posts
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

class SocialSearchState(TypedDict):
    messages:  Annotated[str, add_messages]

def extract_similar_posts(state: SocialSearchState):
    search_model = llm.bind_tools([get_relevant_posts])

    results = search_model.invoke([SystemMessage("""
        Act as the lead social media data scientist at overtime sports
        use the ger_relevant_post tools to extract similar posts with a high
        outlier score. Outlier scores are a way to see how well posts are doing
        relative to average, they signal what does the audience really
        resonate with
    """), *state["messages"]])

    return {"messages": [results]}

execute_tools = ToolNode([get_relevant_posts])

social_search_bot = StateGraph(SocialSearchState)

social_search_bot.add_node("QA", extract_similar_posts)
social_search_bot.add_node("tools", execute_tools)

social_search_bot.add_edge(START, "QA")
social_search_bot.add_conditional_edges("QA", tools_condition)
social_search_bot.add_edge("tools", "QA")

social_app = social_search_bot.compile()

if __name__ == "__main__":
    messages = []
    
    print("Social_Search Agent (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        messages.append(HumanMessage(content=user_input))
        result = social_app.invoke({"messages": messages})
        
        # Update messages with full conversation
        messages = result["messages"]
        
        # Print agent response
        print(f"\nAgent: {messages[-1].content}")