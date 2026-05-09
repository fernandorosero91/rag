"""
================================================================
PRUEBA SIMPLE - RAG ASISTENTE
================================================================
Prueba el sistema RAG sin necesidad de audio.
Escribe una pregunta y obtén respuesta de tus documentos.
================================================================
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("  🧪 PRUEBA DE PREGUNTA - RAG ASISTENTE")
print("="*60)
print()

# ── Cargar dependencias ───────────────────────────────────────
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import requests
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Ejecuta primero: 1_INSTALAR.bat")
    sys.exit(1)

# ── Configuración ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 8))  # Aumentado para más contexto
DB_PATH = "./db"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Cargar RAG ────────────────────────────────────────────────
print("🧠 Cargando modelo de embeddings...")
modelo_embed = SentenceTransformer(EMBED_MODEL)

print("🗄️  Cargando base de datos...")
try:
    cliente = chromadb.PersistentClient(path=DB_PATH)
    coleccion = cliente.get_collection("documentos_rag")
    count = coleccion.count()
    print(f"✅ Base de datos cargada: {count} fragmentos\n")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Ejecuta primero: 2_INDEXAR_PDFS.bat\n")
    sys.exit(1)

# ── Función de búsqueda RAG ───────────────────────────────────
def buscar_en_documentos(pregunta: str):
    """Busca fragmentos relevantes en la base de datos."""
    embedding = modelo_embed.encode([pregunta]).tolist()
    resultados = coleccion.query(
        query_embeddings=embedding,
        n_results=RAG_TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    
    fragmentos = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0]
    ):
        relevancia = 1 - dist
        if relevancia > 0.25:
            fragmentos.append({
                "texto": doc,
                "fuente": meta.get("fuente", "?"),
                "relevancia": relevancia
            })
    
    return fragmentos

# ── Función para llamar LLM ───────────────────────────────────
def obtener_respuesta(pregunta: str, contexto: str):
    """Llama a Groq para obtener respuesta."""
    
    sistema = f"""Eres un asistente experto que responde preguntas usando SOLO la información del contexto dado.

REGLAS:
- Proporciona respuestas COMPLETAS y DETALLADAS usando toda la información disponible en el contexto
- Si el contexto contiene una lista, enumera TODOS los elementos
- Usa formato claro con numeración o viñetas cuando sea apropiado
- Si la información está incompleta en el contexto, menciona qué falta
- Español neutro y profesional
- NO inventes información que no esté en el contexto

CONTEXTO DE LOS DOCUMENTOS:
{contexto}"""
    
    # Intentar Groq primero
    if GROQ_API_KEY and GROQ_API_KEY != "tu_groq_api_key_aqui":
        try:
            print("💬 Consultando Groq...")
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": sistema},
                        {"role": "user", "content": pregunta}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.2
                },
                timeout=10
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip(), "Groq"
        except Exception as e:
            print(f"⚠️  Groq falló: {e}")
    
    # Fallback a OpenRouter
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "tu_openrouter_api_key_aqui":
        try:
            print("💬 Consultando OpenRouter...")
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": sistema},
                        {"role": "user", "content": pregunta}
                    ],
                    "max_tokens": 1000
                },
                timeout=15
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip(), "OpenRouter"
        except Exception as e:
            print(f"⚠️  OpenRouter falló: {e}")
    
    # Fallback a Gemini
    if GEMINI_API_KEY and GEMINI_API_KEY != "tu_gemini_api_key_aqui":
        try:
            print("💬 Consultando Gemini...")
            mensaje = f"{sistema}\n\nPregunta: {pregunta}"
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": mensaje}]}]},
                timeout=15
            )
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip(), "Gemini"
        except Exception as e:
            print(f"⚠️  Gemini falló: {e}")
    
    return "❌ No se pudo obtener respuesta de ninguna API", "Error"

# ── Loop principal ────────────────────────────────────────────
print("="*60)
print("  💡 MODO PRUEBA ACTIVADO")
print("="*60)
print()
print("Escribe tu pregunta sobre el documento indexado.")
print("Escribe 'salir' para terminar.")
print()

while True:
    print("-"*60)
    pregunta = input("❓ Tu pregunta: ").strip()
    
    if not pregunta:
        continue
    
    if pregunta.lower() in ['salir', 'exit', 'quit']:
        print("\n👋 ¡Hasta luego!")
        break
    
    print()
    print("🔍 Buscando en documentos...")
    
    # Buscar en RAG
    fragmentos = buscar_en_documentos(pregunta)
    
    if not fragmentos:
        print("⚠️  No encontré información relevante en los documentos.")
        print()
        continue
    
    print(f"✅ Encontrados {len(fragmentos)} fragmentos relevantes")
    
    # Mostrar fuentes
    fuentes = list(set(f["fuente"] for f in fragmentos))
    print(f"📄 Fuentes: {', '.join(fuentes)}")
    print()
    
    # Construir contexto
    contexto_partes = []
    for i, f in enumerate(fragmentos, 1):
        contexto_partes.append(
            f"[Fuente: {f['fuente']} | Relevancia: {f['relevancia']:.0%}]\n{f['texto']}"
        )
    contexto = "\n\n---\n\n".join(contexto_partes)
    
    # Obtener respuesta
    respuesta, proveedor = obtener_respuesta(pregunta, contexto)
    
    print("="*60)
    print(f"💡 RESPUESTA ({proveedor}):")
    print("="*60)
    print(respuesta)
    print("="*60)
    print()
