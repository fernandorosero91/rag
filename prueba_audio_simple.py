"""
================================================================
PRUEBA DE AUDIO SIMPLE - RAG ASISTENTE
================================================================
Habla por tu micrófono y el sistema responderá automáticamente.
Presiona Ctrl+C para detener.
================================================================
"""

import os
import sys
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("  🎙️ PRUEBA DE AUDIO SIMPLE - RAG ASISTENTE")
print("="*60)
print()

# ── Cargar dependencias ───────────────────────────────────────
try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
    import chromadb
    from sentence_transformers import SentenceTransformer
    import requests
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# ── Configuración ─────────────────────────────────────────────
GROQ_API_KEY_1 = os.getenv("GROQ_API_KEY_1", "")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3", "")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama3.3-70b")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "es")
RAG_TOP_K = 8
SAMPLE_RATE = 16000
RECORD_SECONDS = 5  # Graba 5 segundos cada vez

# ── Cargar modelos ────────────────────────────────────────────
print("🎙️ Cargando Whisper...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

print("🧠 Cargando embeddings...")
embed_model = SentenceTransformer("BAAI/bge-m3")

print("🗄️ Cargando base de datos...")
try:
    cliente = chromadb.PersistentClient(path="./db")
    coleccion = cliente.get_collection("documentos_rag")
    print(f"✅ {coleccion.count()} fragmentos cargados\n")
except Exception as e:
    print(f"❌ Error: {e}\n")
    sys.exit(1)

# ── Funciones ─────────────────────────────────────────────────
def buscar_en_documentos(pregunta: str):
    embedding = embed_model.encode([pregunta]).tolist()
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

def obtener_respuesta(pregunta: str, contexto: str):
    sistema = f"""Eres un asistente experto que responde preguntas usando SOLO la información del contexto dado.

REGLAS:
- Proporciona respuestas COMPLETAS y DETALLADAS usando toda la información disponible
- Si el contexto contiene una lista, enumera TODOS los elementos
- Usa formato claro con numeración cuando sea apropiado
- Español neutro y profesional

CONTEXTO:
{contexto}"""
    
    # Construir lista de proveedores
    proveedores = []
    if GROQ_API_KEY_1 and "tu_groq_api_key" not in GROQ_API_KEY_1:
        proveedores.append(("Groq-1", GROQ_API_KEY_1))
    if GROQ_API_KEY_2 and "tu_groq_api_key" not in GROQ_API_KEY_2:
        proveedores.append(("Groq-2", GROQ_API_KEY_2))
    if GROQ_API_KEY_3 and "tu_groq_api_key" not in GROQ_API_KEY_3:
        proveedores.append(("Groq-3", GROQ_API_KEY_3))
    
    # Intentar Groq en cascada
    for nombre, key in proveedores:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
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
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ {nombre} falló: {e}")
    
    # Fallback a Cerebras
    if CEREBRAS_API_KEY and CEREBRAS_API_KEY != "tu_cerebras_api_key_aqui":
        try:
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
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ Cerebras falló: {e}")
    
    return "No se pudo obtener respuesta"

def detectar_pregunta(texto: str) -> bool:
    """Detecta preguntas de forma tolerante (Whisper puede perder la primera palabra)."""
    texto_lower = texto.lower().strip()
    
    if "?" in texto:
        return True
    
    palabras = texto_lower.split()
    if not palabras:
        return False
    
    interrogativas = [
        "qué", "que", "cómo", "como", "cuál", "cual", "cuáles", "cuales",
        "cuándo", "cuando", "dónde", "donde", "por qué", "quién", "quien",
        "cuánto", "cuanto", "cuántos", "cuántas"
    ]
    
    comandos = [
        "explica", "explique", "describe", "menciona", "define",
        "dime", "dame", "muéstrame", "cuéntame", "enumera", "lista",
        "háblame", "hablame", "habla", "háblanos", "hablanos", "detalla"
    ]
    
    # Buscar interrogativas en las primeras 5 palabras
    primeras_5 = palabras[:5]
    for palabra in primeras_5:
        if palabra in interrogativas:
            return True
    
    # Buscar comandos en las primeras 3 palabras
    primeras_3 = palabras[:3]
    for palabra in primeras_3:
        if palabra in comandos:
            return True
    
    # Frases parciales
    texto_inicio = " ".join(palabras[:6])
    if "por qu" in texto_inicio or "para qu" in texto_inicio:
        return True
    
    # Patrones que indican pregunta aunque falte la primera palabra
    patrones_pregunta = [
        "es el", "son los", "son las", "es la",
        "se basa", "se define", "se establece", "se menciona",
        "significa", "implica", "establece"
    ]
    if len(palabras) >= 4:
        inicio = " ".join(palabras[:3])
        for patron in patrones_pregunta:
            if inicio.startswith(patron):
                return True
    
    return False

# ── Listar dispositivos ───────────────────────────────────────
print("="*60)
print("  🎤 DISPOSITIVOS DE AUDIO DISPONIBLES")
print("="*60)

dispositivos = sd.query_devices()
entradas = []
for i, d in enumerate(dispositivos):
    if d["max_input_channels"] > 0:
        entradas.append((i, d["name"]))
        print(f"  [{i}] {d['name']}")

print()
print("="*60)

# Seleccionar dispositivo
while True:
    try:
        idx = input("Selecciona el número de tu micrófono (Enter para default): ").strip()
        if idx == "":
            device_idx = None
            print("✅ Usando dispositivo por defecto\n")
            break
        device_idx = int(idx)
        if any(i == device_idx for i, _ in entradas):
            print(f"✅ Usando dispositivo [{device_idx}]\n")
            break
        else:
            print("❌ Número inválido, intenta de nuevo")
    except ValueError:
        print("❌ Ingresa un número válido")

# ── Loop principal ────────────────────────────────────────────
print("="*60)
print("  🎙️ MODO AUDIO ACTIVADO")
print("="*60)
print()
print("📢 Instrucciones:")
print("   1. Presiona ENTER para grabar 5 segundos")
print("   2. Habla tu pregunta claramente")
print("   3. El sistema transcribirá y responderá")
print("   4. Escribe 'salir' para terminar")
print()

while True:
    comando = input("Presiona ENTER para grabar (o 'salir' para terminar): ").strip()
    
    if comando.lower() in ['salir', 'exit', 'quit']:
        print("\n👋 ¡Hasta luego!")
        break
    
    print(f"\n🔴 GRABANDO {RECORD_SECONDS} segundos... ¡HABLA AHORA!")
    
    # Grabar audio
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        device=device_idx
    )
    sd.wait()
    
    print("✅ Grabación completada")
    print("🎙️ Transcribiendo...")
    
    # Transcribir
    audio_data = audio[:, 0]
    segmentos, info = whisper_model.transcribe(
        audio_data,
        language=WHISPER_LANGUAGE,
        vad_filter=True
    )
    
    texto = " ".join([s.text for s in segmentos]).strip()
    
    if not texto or len(texto) < 5:
        print("⚠️ No se detectó audio claro. Intenta de nuevo.\n")
        continue
    
    print(f"\n📝 Transcripción: \"{texto}\"")
    
    # Verificar si es pregunta
    if not detectar_pregunta(texto):
        print("ℹ️  No parece ser una pregunta. Intenta de nuevo.\n")
        continue
    
    print("❓ Pregunta detectada!")
    print("🔍 Buscando en documentos...")
    
    # Buscar en RAG
    fragmentos = buscar_en_documentos(texto)
    
    if not fragmentos:
        print("⚠️ No encontré información relevante.\n")
        continue
    
    print(f"✅ Encontrados {len(fragmentos)} fragmentos relevantes")
    
    # Construir contexto
    contexto_partes = []
    for f in fragmentos:
        contexto_partes.append(f"[{f['fuente']}]\n{f['texto']}")
    contexto = "\n\n---\n\n".join(contexto_partes)
    
    print("💬 Consultando IA...")
    
    # Obtener respuesta
    respuesta = obtener_respuesta(texto, contexto)
    
    print("\n" + "="*60)
    print("💡 RESPUESTA:")
    print("="*60)
    print(respuesta)
    print("="*60)
    print()
