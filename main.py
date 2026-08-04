from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents import build_agent
from memory import get_history, save_message, clear_history, get_history_as_dict
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="AI Assistant")


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

agent = build_agent()

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ClearRequest(BaseModel):
    session_id: str = "default"

@app.get("/")
def home():
    return {"message": "AI Assistant API running!"}

@app.post("/ask")
def ask(request: ChatRequest):
    try:
        result = agent.invoke({
            "question": request.question,
            "category": "",
            "answer": "",
            "chat_history": get_history(request.session_id)
        })
        save_message(request.session_id, request.question, result["answer"])
        return {
            "question": request.question,
            "answer": result["answer"],
            "category": result["category"],
            "session_id": request.session_id
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask-stream")
def ask_stream(request: ChatRequest):
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer clearly."),
            ("human", "{question}")
        ]) | llm | StrOutputParser()
    )

    def generate():
        full_response = ""
        try:
            for chunk in chain.stream({"question": request.question}):
                full_response += chunk
                yield chunk
        except Exception as e:
            yield f"\nSorry, something went wrong."
        finally:
            # Save message separately — errors here won't affect the response
            try:
                if full_response:
                    save_message(
                        request.session_id,
                        request.question,
                        full_response
                    )
            except:
                pass   # silently ignore save errors

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/history/{session_id}")
def get_chat_history(session_id: str):
    return {
        "session_id": session_id,
        "history": get_history_as_dict(session_id)
    }

@app.post("/clear")
def clear(request: ClearRequest):
    clear_history(request.session_id)
    return {"message": f"History cleared for {request.session_id}"}

@app.get("/chat", response_class=HTMLResponse)
def chat_ui():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()