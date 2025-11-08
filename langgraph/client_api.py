import requests
from pydantic import BaseModel, ValidationError
from enum import Enum

# ===================================
# 1️⃣ Definizione modelli di output
# ===================================

class TipoPersona(str, Enum):
    fisica = "fisica"
    giuridica = "giuridica"

class PersonaOutput(BaseModel):
    tipo: TipoPersona

# ===================================
# 2️⃣ Funzione client con gestione errori
# ===================================

def classifica_persona_api(nome: str, base_url: str = "http://127.0.0.1:8000") -> PersonaOutput:
    """
    Interroga l'API FastAPI /classifica e restituisce il risultato tipizzato.
    Gestisce errori di connessione e validazione.
    """
    url = f"{base_url}/classifica"
    payload = {"nome": nome}

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()  # solleva errore HTTP se 4xx o 5xx

        # valida la risposta con Pydantic
        result = PersonaOutput(**response.json())
        return result

    except requests.exceptions.ConnectionError:
        print("❌ Errore: impossibile connettersi al server FastAPI. È in esecuzione?")
    except requests.exceptions.Timeout:
        print("⚠️ Timeout: il server non ha risposto entro 10 secondi.")
    except requests.exceptions.HTTPError as e:
        print(f"❌ Errore HTTP: {e} - Dettagli: {response.text}")
    except ValidationError as e:
        print(f"⚠️ Errore di validazione nel formato della risposta: {e}")
    except Exception as e:
        print(f"⚠️ Errore imprevisto: {e}")

    return None  # in caso di errore

# ===================================
# 3️⃣ Esempio d’uso
# ===================================

if __name__ == "__main__":
    nome_input = "Mario Rossi"
    result = classifica_persona_api(nome_input)

    if result:
        print(f"✅ '{nome_input}' è una persona {result.tipo.value}")
    else:
        print("❌ Nessun risultato ottenuto.")
