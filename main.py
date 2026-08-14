from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ Error 1 fixed
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from agents import build_agent
from memory import get_history, save_message, clear_history, get_history_as_dict
import os
import base64                                                          # ✅ Error 6 fixed
import requests as req
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Assistant")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

# ✅ Error 17 fixed — groq client defined
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

HF_TOKEN = os.environ.get("HF_TOKEN")

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
            try:
                if full_response:
                    save_message(
                        request.session_id,
                        request.question,
                        full_response
                    )
            except:
                pass

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

# ---- PDF Upload ----
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(f"uploads/{file.filename}")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100                    # ✅ Error 2 fixed
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("pdf_index")

    return {"message": f"PDF processed! {len(chunks)} chunks created"}

# ---- Ask PDF ----
@app.post("/ask-pdf")
def ask_pdf(request: ChatRequest):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "pdf_index",
        embeddings,                          # ✅ Error 3 fixed — comma added
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}               # ✅ Error 4 fixed — colon not comma
    )
    docs = retriever.invoke(request.question)
    context = "\n".join([d.page_content for d in docs])

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Answer using this context:\n{context}"),  # ✅ Error 18 fixed
            ("human", "{question}")
        ]) | llm | StrOutputParser()
    )
    answer = chain.invoke({
        "context": context,
        "question": request.question
    })
    return {"answer": answer}               # ✅ Error 5 fixed — {} not ()

# ---- Analyze Image ----
@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    question: str = "What is in this image? Describe in detail."
):
    # Add formatting instruction to every question
    formatted_question = f"""{question}

Please format your response as:
- Use bullet points for lists
- Use short clear sentences
- Add headers for different sections
- Keep paragraphs short (2-3 lines max)"""
    try:
        import time
        from google import genai as google_genai
        from google.genai import types

        client = google_genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
        )

        image_data = await file.read()
        content_type = file.content_type or "image/jpeg"

        # Only use models with good free quota
        models_to_try = [
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-flash-latest",
        ]

        last_error = ""
        for model_name in models_to_try:
            try:
                print(f"Trying: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(
                            data=image_data,
                            mime_type=content_type
                        ),
                        question
                    ]
                )
                print(f"✅ Success with: {model_name}")
                return {
                    "success": True,
                    "question": question,
                    "answer": response.text
                }
            except Exception as e:
                last_error = str(e)
                print(f"❌ Failed {model_name}: {last_error[:100]}")
                time.sleep(1)
                continue

        return {
            "success": False,
            "error": "Image analysis temporarily unavailable. Try again in a few minutes."
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
#------Celelrbrity photos -------
@app.get("/celebrity-photo/{name}")
def get_celebrity_photo(name: str):
    try:
        # Capitalize each word properly
        proper_name = name.strip().title()

        # Try direct Wikipedia API
        response = req.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" +
            proper_name.replace(" ", "_"),
            headers={"User-Agent": "AIAssistant/1.0"}
        )
        data = response.json()

        # If no thumbnail try search API
        if "thumbnail" not in data:
            search_response = req.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": proper_name,
                    "format": "json",
                    "srlimit": 1
                }
            )
            search_data = search_response.json()
            results = search_data.get("query", {}).get("search", [])

            if results:
                page_title = results[0]["title"]
                # Fetch that page
                response = req.get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/" +
                    page_title.replace(" ", "_"),
                    headers={"User-Agent": "AIAssistant/1.0"}
                )
                data = response.json()

        if "thumbnail" in data:
            return {
                "name": data.get("title", proper_name),
                "photo": data["thumbnail"]["source"],
                "description": data.get("extract", "")[:300],
                "wikipedia": data.get("content_urls", {})
                              .get("desktop", {}).get("page", "")
            }
        else:
            return {
                "name": proper_name,
                "photo": None,
                "description": data.get("extract", "No info found")[:300],
                "wikipedia": data.get("content_urls", {})
                              .get("desktop", {}).get("page", "")
            }
    except Exception as e:
        return {"error": str(e), "photo": None, "name": name}

# ---- Generate Image ----
@app.post("/generate-image")
async def generate_image(request: ChatRequest):
    try:
        # Enhance prompt using LLM
        enhance_chain = (
            ChatPromptTemplate.from_messages([
                ("system", """You are an expert at writing image generation prompts.
Make the description detailed and vivid. Keep under 100 words."""),
                ("human", "{prompt}")
            ]) | llm | StrOutputParser()
        )

        enhanced_prompt = enhance_chain.invoke({
            "prompt": request.question
        })

        print(f"Enhanced prompt: {enhanced_prompt}")

        # Pollinations AI — FREE, no token needed!
        import urllib.parse
        encoded_prompt = urllib.parse.quote(enhanced_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

        print(f"Fetching image from: {image_url}")

        response = req.get(image_url, timeout=60)

        if response.status_code == 200:
            return Response(
                content=response.content,
                media_type="image/jpeg"
            )
        else:
            return {
                "success": False,
                "error": f"Failed with status: {response.status_code}"
            }

    except Exception as e:
        return {"success": False, "error": str(e)}

UNSPLASH_KEY = os.environ.get("UNSPLASH_ACCESS_KEY")

@app.get("/search-photos/{query}")
def search_photos(query: str):
    try:
        response = req.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": query,
                "per_page": 12,
                "orientation": "landscape"
            },
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"}
        )

        data = response.json()

        # Check if results exist
        if "results" not in data:
            return {"photos": [], "query": query, "error": str(data)}

        photos = []
        for photo in data["results"]:
            try:
                photos.append({
                    "url": photo["urls"]["regular"],
                    "small": photo["urls"]["small"],
                    "description": photo.get("alt_description") or query,
                    "photographer": photo["user"]["name"],
                    "photographer_url": photo["user"]["links"]["html"]
                })
            except Exception:
                continue

        return {
            "photos": photos,
            "query": query,
            "total": len(photos)
        }

    except Exception as e:
        return {"error": str(e), "photos": []}
# ---- Weather ----
@app.get("/weather/{city}")
def get_weather(city: str):
    import requests
    url = f"https://wttr.in/{city}?format=j1"            # ✅ Error 11 fixed — / added
    data = requests.get(url).json()                      # ✅ Error 12 fixed
    temp = data['current_condition'][0]['temp_C']
    desc = data['current_condition'][0]['weatherDesc'][0]['value']

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a weather assistant."),
            ("human", f"Weather in {city}: {temp}°C, {desc}. Give advice.")  # ✅ Error 13 fixed
        ]) | llm | StrOutputParser()
    )
    advice = chain.invoke({})
    return {
        "city": city,
        "temperature": f"{temp}°C",
        "description": desc,
        "advice": advice
    }

# ---- Wikipedia ----
@app.get("/wiki/{topic}")
def search_wikipedia(topic: str):
    import wikipediaapi
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(topic)

    if page.exists():
        summary = page.summary[:1000]
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "Summarize this Wikipedia content simply."),
                ("human", "{text}")
            ]) | llm | StrOutputParser()
        )
        result = chain.invoke({"text": summary})
        return {
            "topic": topic,
            "summary": result
        }
    return {"error": "Topic not found"}

# ---- Admin Dashboard ----
@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard():
    from memory import sessions
    total_sessions = len(sessions)
    total_messages = sum(len(v) for v in sessions.values())

    return f"""
    <html>
    <body style="font-family:Arial;padding:20px">
        <h1>Admin Dashboard</h1>
        <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:20px">
            <div style="background:#007bff;color:white;padding:20px;border-radius:8px">
                <h2>{total_sessions}</h2>
                <p>Active Sessions</p>
            </div>
            <div style="background:#28a745;color:white;padding:20px;border-radius:8px">
                <h2>{total_messages}</h2>
                <p>Total Messages</p>
            </div>
        </div>
        <br>
        <a href="/history/default">View History</a>
    </body>
    </html>
    """

from fastapi.staticfiles import StaticFiles
import shutil

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- Upload your own photo ----
@app.post("/upload-my-photo")
async def upload_my_photo(
    file: UploadFile = File(...),
    name: str = "My Name",
    description: str = "About me"
):
    try:
        os.makedirs("static", exist_ok=True)

        # Save file
        filename = file.filename
        filepath = f"static/{filename}"
        with open(filepath, "wb") as f:
            f.write(await file.read())

        # Save info to a JSON file
        import json
        custom_celebs = []

        if os.path.exists("static/custom_celebs.json"):
            with open("static/custom_celebs.json", "r") as f:
                custom_celebs = json.load(f)

        # Add new entry
        custom_celebs.append({
            "name": name,
            "photo": f"/static/{filename}",
            "description": description,
            "wikipedia": ""
        })

        with open("static/custom_celebs.json", "w") as f:
            json.dump(custom_celebs, f)

        return {
            "success": True,
            "message": f"Photo uploaded for {name}!",
            "photo_url": f"/static/{filename}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ---- Get custom photos ----
@app.get("/my-celebs")
def get_my_celebs():
    import json
    try:
        if os.path.exists("static/custom_celebs.json"):
            with open("static/custom_celebs.json", "r") as f:
                return {"celebs": json.load(f)}
        return {"celebs": []}
    except Exception as e:
        return {"error": str(e), "celebs": []}