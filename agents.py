from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="groq/compound",
    api_key=os.environ.get("GROQ_API_KEY")
)

class AgentState(TypedDict):
    question: str
    category: str
    answer: str
    chat_history: list

# Agent1 classifier

def classifier_agent(state: AgentState) -> AgentState:
    print("classifying question")
    chain= (ChatPromptTemplate.from_messages([
        ("system", """Classify the question into exactly one word.
Reply ONLY with: general, math, science, coding"""),
("human", "{question}")
    ]) | llm | StrOutputParser()
    )
    category = chain.invoke({"question" : state["question"]})
    category= category.strip().lower()


    if "math" in category:
        category= "math"

    elif "science" in category:
        category= "science"

    elif "coding" in category:
        category= "coding"

    else:
        category= "general"

    print(f"category: {category}")
    return {"category": category}


def general_agent(state: AgentState)-> AgentState:
    print("General agent working")
    chain= (ChatPromptTemplate.from_messages([
     ("system","You are helpful friendly assistant. Answer clearly"),
         ("human","{question}")
    ]) | llm | StrOutputParser()
    ) 

    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer}

def math_agent(state: AgentState)-> AgentState:
    print("General agent working")
    chain= (ChatPromptTemplate.from_messages([
     ("system","You are mathematics expert. Give step by step solution"),
         ("human","{question}")
    ]) | llm | StrOutputParser()
    ) 

    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer}

def science_agent(state: AgentState)-> AgentState:
    print("General agent working")
    chain= (ChatPromptTemplate.from_messages([
     ("system","You are science expert. Explain with examples"),
         ("human","{question}")
    ]) | llm | StrOutputParser()
    ) 

    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer}


def coding_agent(state: AgentState)-> AgentState:
    print("General agent working")
    chain= (ChatPromptTemplate.from_messages([
     ("system","You are an expert Programmer. please provide full working code with examples"),
         ("human","{question}")
    ]) | llm | StrOutputParser()
    ) 

    answer = chain.invoke({"question": state["question"]})
    return {"answer": answer}

def route_agent(state: AgentState)-> AgentState:
    return state["category"]


def build_agent():
    graph= StateGraph(AgentState)

    graph.add_node("classifier", classifier_agent)
    graph.add_node("general", general_agent)
    graph.add_node("math", math_agent)
    graph.add_node("science", science_agent)
    graph.add_node("coding", coding_agent)

    graph.set_entry_point("classifier")

    graph.add_conditional_edges(
        "classifier",
        route_agent,
        {
            "general": "general",
            "math": "math",
            "science": "science",
            "coding":"coding"
        }
    )

    graph.add_edge("general",END)
    graph.add_edge("math",END)
    graph.add_edge("science",END)
    graph.add_edge("coding",END)

    return graph.compile()

    