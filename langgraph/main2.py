from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
from langgraph.graph import Graph

# ======================================================
# 1️⃣ MODELLI
# ======================================================

class TipoPersona(str, Enum):
    fisica = "fisica"
    giuridica = "giuridica"

class PersonaOutput(BaseModel):
    tipo: TipoPersona = Field(description="Tipo di persona: 'fisica' o 'giuridica'")

# modello intermedio (solo per parsing nome/cognome)
class DettaglioNomeCognome(BaseModel):
    nome: str
    cognome: str

# modelli finali (output strutturato)
class DettaglioPersonaFisica(DettaglioNomeCognome):
    tipo: TipoPersona = Field(default=TipoPersona.fisica)

class DettaglioPersonaGiuridica(BaseModel):
    ragione_sociale: str
    tipo: TipoPersona = Field(default=TipoPersona.giuridica)

# ======================================================
# 2️⃣ LLM
# ======================================================

llm = ChatOllama(model="llama3.2", temperature=0)

# ======================================================
# 3️⃣ NODI DEL GRAFO
# ======================================================

def classifica_persona(input_str: str):
    prompt = f"""
    Determina se il seguente nome si riferisce a una persona fisica o giuridica.
    Rispondi solo con 'fisica' o 'giuridica'.

    Nome: {input_str}
    """
    result = llm.with_structured_output(PersonaOutput).invoke([HumanMessage(content=prompt)])
    return {"input_str": input_str, "classificazione": result}

def elabora_dettagli(data: dict):
    classificazione = data["classificazione"]
    input_str = data["input_str"]

    if classificazione.tipo == TipoPersona.fisica:
        prompt = f"""
        Dividi il seguente nome in nome e cognome.
        Rispondi in formato JSON con le chiavi 'nome' e 'cognome'.

        Nome completo: {input_str}
        """
        parsed = llm.with_structured_output(DettaglioNomeCognome).invoke([HumanMessage(content=prompt)])
        return DettaglioPersonaFisica(**parsed.model_dump())
    else:
        return DettaglioPersonaGiuridica(ragione_sociale=input_str)

# ======================================================
# 4️⃣ COSTRUZIONE GRAFO
# ======================================================

graph = Graph()
graph.add_node("classifica", classifica_persona)
graph.add_node("elabora_dettagli", elabora_dettagli)
graph.add_edge("classifica", "elabora_dettagli")
graph.set_entry_point("classifica")
graph.set_finish_point("elabora_dettagli")
app_graph = graph.compile()

# ======================================================
# 5️⃣ FASTAPI APP
# ======================================================

app = FastAPI(title="Persona Classifier API", version="1.0")

class InputRequest(BaseModel):
    input_text: str = Field(..., description="Nome o ragione sociale da analizzare")

@app.post("/classifica")
async def classifica_api(req: InputRequest):
    """
    Endpoint REST che riceve una stringa e restituisce
    un output strutturato: persona fisica (nome/cognome)
    oppure persona giuridica (ragione sociale).
    """
    result = app_graph.invoke(req.input_text)
    return result.model_dump()

# ======================================================
# 6️⃣ AVVIO SERVER
# ======================================================
# Esegui con:  uvicorn main:app --reload
