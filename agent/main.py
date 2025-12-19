from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from graph import sf_app
from rfp_graph import rfp_app
from langchain_core.messages import HumanMessage
import PyPDF2
import io

app = FastAPI()

conversation_history = []
rfp_conversation_history = []

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
def generate_rfp(request: RFPRequest):
    """Generate RFP response using the RFP graph"""
    try:
        # The RFP graph expects {"raw_rfp": text} and generates a full proposal
        result = rfp_app.invoke({"raw_rfp": request.pdf_text})

        # Return the final proposal
        return ChatResponse(response=result["final_proposal"])
    except Exception as e:
        return ChatResponse(response=f"Error generating RFP response: {str(e)}")

@app.post("/rfp/clear")
def clear_rfp():
    rfp_conversation_history.clear()
    return {"status": "cleared"}