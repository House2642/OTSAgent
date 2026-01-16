from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from graph import sf_app
from rfp_graph import rfp_app
from brand_insights.brand_insights_agents import brand_insights_agent
from langchain_core.messages import HumanMessage
import PyPDF2
import io
import json
import asyncio

app = FastAPI()

# Mount static files directory for serving logos
app.mount("/static", StaticFiles(directory="static"), name="static")

conversation_history = []
rfp_conversation_history = []
brand_insights_conversation_history = []

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class RFPRequest(BaseModel):
    pdf_text: str
    message: str

# Landing page
@app.get("/", response_class=HTMLResponse)
def landing():
    with open("landing.html") as f:
        return f.read()

# Salesforce bot
@app.get("/salesforce", response_class=HTMLResponse)
def salesforce():
    with open("salesforce.html") as f:
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

# RFP generator
@app.get("/rfp", response_class=HTMLResponse)
def rfp():
    with open("rfp.html") as f:
        return f.read()

@app.post("/rfp/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Extract text from uploaded PDF"""
    try:
        contents = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        return {"text": text, "filename": file.filename}
    except Exception as e:
        return {"error": str(e)}

@app.post("/rfp/generate")
async def generate_rfp(request: RFPRequest):
    """Generate RFP response using the RFP graph with streaming updates"""

    async def event_generator():
        try:
            # Send initial status
            yield f"data: {json.dumps({'stage': 'Starting RFP analysis...', 'progress': 0})}\n\n"
            await asyncio.sleep(0.1)

            # Stage 1: Extracting requirements
            yield f"data: {json.dumps({'stage': 'Extracting requirements from RFP...', 'progress': 20})}\n\n"
            await asyncio.sleep(0.5)

            # Stage 2: Searching past deals
            yield f"data: {json.dumps({'stage': 'Searching past deals and client history...', 'progress': 40})}\n\n"
            await asyncio.sleep(0.5)

            # Stage 3: Gathering audience statistics
            yield f"data: {json.dumps({'stage': 'Gathering audience statistics and insights...', 'progress': 60})}\n\n"
            await asyncio.sleep(0.5)

            # Stage 4: Generating content ideas
            yield f"data: {json.dumps({'stage': 'Generating content ideas and product recommendations...', 'progress': 80})}\n\n"
            await asyncio.sleep(0.5)

            # Stage 5: Creating proposal
            yield f"data: {json.dumps({'stage': 'Creating final proposal document...', 'progress': 90})}\n\n"

            # Run the actual RFP generation
            result = await asyncio.to_thread(rfp_app.invoke, {"raw_rfp": request.pdf_text})

            # Send completion
            yield f"data: {json.dumps({'stage': 'Complete!', 'progress': 100, 'response': result['final_proposal']})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': f'Error generating RFP response: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/rfp/clear")
def clear_rfp():
    rfp_conversation_history.clear()
    return {"status": "cleared"}

# Brand Fam Agent (Brand Insights)
@app.get("/brand-insights", response_class=HTMLResponse)
def brand_insights():
    with open("brand_insights.html") as f:
        return f.read()

@app.post("/brand-insights/chat")
def brand_insights_chat(request: ChatRequest):
    brand_insights_conversation_history.append(HumanMessage(content=request.message))

    result = brand_insights_agent.invoke({"messages": brand_insights_conversation_history})

    # Update history with full conversation
    brand_insights_conversation_history.clear()
    brand_insights_conversation_history.extend(result["messages"])

    answer = result["messages"][-1].content
    return ChatResponse(response=answer)

@app.post("/brand-insights/clear")
def clear_brand_insights():
    brand_insights_conversation_history.clear()
    return {"status": "cleared"}