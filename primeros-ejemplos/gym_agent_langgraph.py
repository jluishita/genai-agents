from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate 
from langchain_google_genai import ChatGoogleGenerativeAI

"""
En este fichero generamos un agente con un flujo hardcodeado escrito en LangGraph

Esto limita la estructura de la conversación, pero es muy útil para mantener un flujo concreto en situaciones
en las que no es admisible salirse de un guión predefinido.

Para usar un agente definido en LangGraph necesitamos definir el estado del sistema, State. Éste almacena:
- Una lista en la que se contienen todos los mensajes que ha habido en la conversación
- Las variables que queramos que el agente extraiga de la interacción con el usuario (en este caso nombre e email)

Definimos estructuras de Pydantic para la extracción del nombre y el email. Pydantic es una forma de garantizar que 
una variable tenga un formato determinado. En caso de no tener ese formato, saltaría un error (que haría falta gestionar.
En este código no se gentionan los errores, por ahora)

Definimos los agentes necesarios y un prompt anti alucinaciones

Definimos tres nodos:
- Dos de ellos extraen información de la conversación: el nombre y el email
- Uno de ellos valida el email. Este nodo se puede considerar como una Tool. En caso de que el email no sea válido,
se retornará al nodo que solicita el email

Definimos el grafo de LangGraph y sobre él:
- Añadimos los nodos
- Añadimos las aristas condicionales. Dependiendo de la información contenida en el State, se llevará al usuario a un nodo
o a otro

Cuando toda la información este recabada y tenga el formato correcto, se detendrá el agente.
"""

# 1. The State (Remains the same)
class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str | None
    email: str | None

# 2. Pydantic structures (Only for extracting "raw" data)
class NameExtraction(BaseModel):
    name: str | None = Field(description="The user's name. Empty (null) if there is no name in the text.")

class EmailExtraction(BaseModel):
    email: str | None = Field(description="The user's email address. Empty (null) if there is no email.")


# 3. Configure the models
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    max_retries=5,
    temperature=0.7) # Slightly higher temp for creative speaking

llm_extractor = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    max_retries=5,
    temperature=0.0) # Slightly higher temp for creative speaking

anti_hallucination_prompt = ChatPromptTemplate.from_messages([ 
    ("system", "You are an exact data extractor. Extract the requested data from the text. " \
    "CRITICAL RULE: If the exact data is NOT explicitly written by the user in the text, you MUST return null/empty. " \
    "DO NOT guess, DO NOT invent, DO NOT assume."), 
    ("human", "{user_text}") ]) 

llm_name_extractor = anti_hallucination_prompt | llm_extractor.with_structured_output(NameExtraction)
llm_email_extractor = anti_hallucination_prompt | llm_extractor.with_structured_output(EmailExtraction)

# --- NODE DEFINITIONS WITH DYNAMIC GENERATION ---

def ask_name_node(state: State):
    messages = state.get("messages", [])
    
    # 1. Try to extract the name if the last message is from the user
    if messages and isinstance(messages[-1], HumanMessage):
        result = llm_name_extractor.invoke(messages[-1].content)
        if result.name:
            # Data acquired! Update state and pass the turn to the next node.
            # We don't generate a message here so the next node can speak.
            return {"name": result.name}

    # 2. If we reach here, the name is missing. Use the LLM to generate the prompt.
    instructions = SystemMessage(content="You are the virtual receptionist of a gym. " \
    "Your goal now is to find out the user's name. " \
    "Greet them and ask for their name naturally and very politely. Be friendly.")
    dynamic_response = llm.invoke([instructions] + messages)
    
    return {"messages": [dynamic_response]}


def ask_email_node(state: State):
    messages = state.get("messages", [])
    
    # 1. Try to extract the email
    if messages and isinstance(messages[-1], HumanMessage):
        result = llm_email_extractor.invoke(messages[-1].content)
        if result.email:
            return {"email": result.email}

    # 2. If the email is missing, generate the prompt using the context we already have
    name = state.get("name")
    instructions = SystemMessage(content=f"You know the user's name is {name}." \
    "Thank them for giving you their name and politely ask for their email address to continue the registration. " \
    "Be friendly.")
    dynamic_response = llm.invoke([instructions] + messages)
    
    return {"messages": [dynamic_response]}


def validate_email_node(state: State):
    email_to_validate = state["email"]
    name = state.get("name")
    print(f"\n[⚙️ VALIDATION NODE] Internal check: {email_to_validate}...")
    
    # Validation logic
    if "@" in email_to_validate and "." in email_to_validate:
        # Correct email: LLM generates the triumphant farewell
        instructions = SystemMessage(content=f"The email '{email_to_validate}' from {name} is VALID and the registration is complete. Say goodbye by giving them an official welcome to the gym in a very enthusiastic and motivating way.")
        dynamic_response = llm.invoke([instructions] + state["messages"])
        return {"messages": [dynamic_response]}
    else:
        # Incorrect email: LLM explains the error politely
        instructions = SystemMessage(content=f"The email '{email_to_validate}' is INVALID (missing an @ or a dot). Explain the error to {name} politely and ask them to type it again.")
        dynamic_response = llm.invoke([instructions] + state["messages"])
        
        # Clear the email from the state to force the graph to go back
        return {"email": None, "messages": [dynamic_response]}


# --- ROUTERS (Graph rules remain identical) ---

def main_router(state: State) -> str:
    if not state.get("name"): return "ask_name_node"
    elif not state.get("email"): return "ask_email_node"
    else: return "validate_email_node"

def post_name_router(state: State) -> str:
    return "ask_email_node" if state.get("name") else "__end__"

def post_email_router(state: State) -> str:
    return "validate_email_node" if state.get("email") else "__end__"

# --- GRAPH CONSTRUCTION ---
builder = StateGraph(State)

builder.add_node("ask_name_node", ask_name_node)
builder.add_node("ask_email_node", ask_email_node)
builder.add_node("validate_email_node", validate_email_node)

builder.add_conditional_edges(START, main_router)
builder.add_conditional_edges("ask_name_node", post_name_router)
builder.add_conditional_edges("ask_email_node", post_email_router)
builder.add_edge("validate_email_node", END)

graph = builder.compile()

# --- TESTING LOOP ---
if __name__ == "__main__":
    # SOLUCIÓN: Añadimos un mensaje inicial para que el LLM tenga un contexto al que responder
    initial_message = HumanMessage(content="Hello, I just arrived at the gym's website.")
    state = {"messages": [initial_message], "name": None, "email": None}
    
    print("--- Hybrid Registration (Strict Flow + Natural Language) ---")
    # ... (el resto del bucle while se queda igual)
    while True:
        for event in graph.stream(state, stream_mode="updates"):
            for node, data in event.items():
                if "messages" in data:
                    print(f"🤖 Receptionist: {data['messages'][-1].content[0]['text']}")
                
                if "name" in data: state["name"] = data["name"]
                if "email" in data: state["email"] = data["email"]
                if "messages" in data: state["messages"].extend(data["messages"])
        
        if state.get("name") and state.get("email"):
            break
             
        user_input = input("\nYou: ")
        state["messages"].append(HumanMessage(content=user_input))
