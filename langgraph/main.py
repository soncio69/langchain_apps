# Da eseguire da prompt
# uvicorn main:app --reload

# Da browser : http://127.0.0.1:8000/docs
# Try it out

from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from langgraph.graph import Graph
import uvicorn

# ===============================
# 1️⃣ MODELLI Pydantic
# ===============================

class TipoPersona(str, Enum):
    fisica = "fisica"
    giuridica = "giuridica"

class PersonaOutput(BaseModel):
    tipo: TipoPersona = Field(description="Tipo di persona: 'fisica' o 'giuridica'")

class InputData(BaseModel):
    nome: str

# ===============================
# 2️⃣ MODELLO LLM
# ===============================

llm = ChatOllama(model="llama3.2", temperature=0)

# ===============================
# 3️⃣ NODI DEL GRAFO
# ===============================

def classifica_persona(input_str: str) -> PersonaOutput:
    prompt = f"""
    Determina se il seguente nome si riferisce a una persona fisica o giuridica.
    Rispondi solo con 'fisica' o 'giuridica'.

    Nome: {input_str}
    """
    response = llm.with_structured_output(PersonaOutput).invoke([HumanMessage(content=prompt)])
    return response

def log_result(result: PersonaOutput) -> PersonaOutput:
    print(f"[LOG] Tipo rilevato: {result.tipo}")
    return result

# ===============================
# 4️⃣ COSTRUZIONE DEL GRAFO
# ===============================

graph = Graph()
graph.add_node("classifica", classifica_persona)
graph.add_node("log_result", log_result)
graph.add_edge("classifica", "log_result")
graph.set_entry_point("classifica")
graph.set_finish_point("log_result")

# Compilazione in runnable
app_graph = graph.compile()

# ===============================
# 5️⃣ CREAZIONE API FASTAPI
# ===============================

app = FastAPI(
    title="Classifier API",
    description="API che classifica una persona come fisica o giuridica usando LangGraph + ChatOllama",
    version="1.0.0"
)

@app.post("/classifica", response_model=PersonaOutput)
async def classifica(data: InputData):
    """Endpoint che classifica la persona."""
    result = app_graph.invoke(data.nome)
    return result

# ===============================
# 6️⃣ AVVIO LOCALE
# ===============================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
