from langchain_core.messages import HumanMessage , AIMessage

# dtore session in memory
session={}

def get_history(session_id: str) -> list:
    if session_id not in session:
        session[session_id] =[]
    return session[session_id]

def save_message(session_id: str , question: str , answer:str):
    if session_id not in session: 
        session[session_id]
    session[session_id].append(HumanMessage(Content= question))
    session[session_id].append(AIMessage(content=answer))

def clear_history(session_id: str):
    if session_id in session:
        del session[session_id]

def get_history_as_dict(session_id: str) -> list:
    history = get_history(session_id)
    result=[]
    for msg in history:
        role= "user" if isinstance (msg,HumanMessage) else "assistant"
        result.append({"role": role , "content":msg.content})
    return result