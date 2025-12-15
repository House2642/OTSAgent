from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from graph import sf_app
from langchain_core.messages import HumanMessage

app = FastAPI()

conversation_history = []

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.get("/", response_class=HTMLResponse)
def home():
    with open("frontend.html") as f:
        return f.read()

@app.post("/chat")
def chat(request: ChatRequest):
    conversation_history.append(HumanMessage(content=request.message))
    
    result = sf_app.invoke({"messages": conversation_history})
    
    # Update history with full conversation
    conversation_history.clear()
    conversation_history.extend(result["messages"])
    
    answer = result["messages"][-1].content
    return ChatResponse(response=answer)

@app.post("/clear")
def clear():
    conversation_history.clear()
    return {"status": "cleared"}