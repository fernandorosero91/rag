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
GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1", "")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.3-70b")
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
def _llamar_groq(api_key, pregunta, sistema):
    """Llama a Groq con una key específica."""
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
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
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def obtener_respuesta(pregunta: str, contexto: str):
    """Llama a los proveedores en cascada: Groq1 → Groq2 → Groq3 → Cerebras."""
    
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
    
    # Construir lista de proveedores disponibles
    proveedores = []
    if GROQ_API_KEY_1 and "tu_groq_api_key" not in GROQ_API_KEY_1:
        proveedores.append(("Groq-1", lambda: _llamar_groq(GROQ_API_KEY_1, pregunta, sistema)))
    if GROQ_API_KEY_2 and "tu_groq_api_key" not in GROQ_API_KEY_2:
        proveedores.append(("Groq-2", lambda: _llamar_groq(GROQ_API_KEY_2, pregunta, sistema)))
    if GROQ_API_KEY_3 and "tu_groq_api_key" not in GROQ_API_KEY_3:
        proveedores.append(("Groq-3", lambda: _llamar_groq(GROQ_API_KEY_3, pregunta, sistema)))
    if CEREBRAS_API_KEY and CEREBRAS_API_KEY != "tu_cerebras_api_key_aqui":
        def _llamar_cerebras():
            resp = requests.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": CEREBRAS_MODEL,
                    "messages": [
                        {"role": "system", "content": sistema},
                        {"role": "user", "content": pregunta}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.2
                },
                timeout=12
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        proveedores.append(("Cerebras", _llamar_cerebras))
    
    for nombre, fn in proveedores:
        try:
            print(f"💬 Consultando {nombre}...")
            respuesta = fn()
            return respuesta, nombre
        except Exception as e:
            print(f"⚠️  {nombre} falló: {e}")
    
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
