from typing import TypedDict, Annotated
import operator
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")  # Fixed: model= keyword

from .demographics_tools import account_search, fan_overlap, sports_fandom, audience_demographics

DEBUG = True
tools = [account_search, fan_overlap, sports_fandom, audience_demographics]
model = llm.bind_tools(tools)

class demographics_sub_agent(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

def demographics_retrieval(state: demographics_sub_agent):
    response = model.invoke([SystemMessage(content="""  # Fixed: ( instead of [, added content=
        You are a data analytics agent for Overtime sports with access to data on the overtime accounts, and 
        demographics about the overtime audience. 
        
        TOOLS:
            account_search: You can search accounts and specific categories of statistics about it
            fan_overlap: search the overlap of fandom across popular leagues like NFL and NBA for overtime fans
            sports_fandom: which sports and leagues the overtime audience are avid fans of, they watch live and stream live, and which
            sports they currently play + sports they want to play, an ot index > 150 means it's a significant over index
            audience_demographics: Use this tool to understand the ethinicity, household income, employment, living situation, education, language, and relationshipstatus/children 
            of the overtime sports audience
        RULE:
            Note LDA Compliance is required for alcohol compliance. This means that at least 71.6% 
            of the audience on that account is 21+. Break out the accounts and channel for example, Overtime Main Instagram is LDA compliant.
            Never output the whole account always the account and platform

        QUESTION EXAMPLES:
            "What account has the most followers on snapchat?"
            "How many over 45+ followers do we have on overtime main?"
            "We are working to promote a product to NFL fans"
    """), *state["messages"]])  # Fixed: ] instead of )
    return {"messages": [response]}

########Create graph#########
execute_tools = ToolNode(tools)
demo_graph = StateGraph(demographics_sub_agent)
demo_graph.add_node("QA", demographics_retrieval)
demo_graph.add_node("tools", execute_tools)

demo_graph.add_edge(START, "QA")
demo_graph.add_conditional_edges("QA", tools_condition)
demo_graph.add_edge("tools", "QA")

demographics_agent = demo_graph.compile()

if __name__ == "__main__":  # Fixed: 3 underscores to 2
    messages = []
    
    print("I am a overtime sports demographics search agent, ask me about the accounts")
    query = input("User: ")
    messages.append(HumanMessage(query))
    
    response = demographics_agent.invoke({
        "messages": messages
    })
    if DEBUG:
        for message in response["messages"]:
            print(message.content)
    print(response["messages"][-1].content)