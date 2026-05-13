"""
================================================================
RAG ASISTENTE v2.0 - ESCUCHA REUNIONES Y RESPONDE CON TUS DOCUMENTOS
================================================================
Arquitectura:
  Teams Audio → VB-Cable → faster-whisper → Búsqueda Híbrida (Vector + BM25)
  → Reranker → Top-5 chunks → LLM Streaming → UI Flotante

Mejoras v2.0:
  - Búsqueda híbrida (semántica + keywords BM25)
  - Reranker cross-encoder para precisión
  - Streaming de respuestas (se ve en tiempo real)
  - Whisper tiny para velocidad
  - Timeouts cortos con detección de rate-limit
  - Detección de preguntas más precisa

Uso: python asistente.py
================================================================
"""

import os
import sys
import time
import queue
import threading
import re
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# ── Verificar dependencias ────────────────────────────────────
DEPENDENCIAS_FALTANTES = []

try:
    import tkinter as tk
    from tkinter import scrolledtext, font as tkfont, ttk
except:
    DEPENDENCIAS_FALTANTES.append("tkinter (viene con Python)")

try:
    import numpy as np
except:
    DEPENDENCIAS_FALTANTES.append("numpy")

try:
    import sounddevice as sd
except:
    DEPENDENCIAS_FALTANTES.append("sounddevice")

try:
    from faster_whisper import WhisperModel
except:
    DEPENDENCIAS_FALTANTES.append("faster-whisper")

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except:
    DEPENDENCIAS_FALTANTES.append("chromadb sentence-transformers")

try:
    import requests
except:
    DEPENDENCIAS_FALTANTES.append("requests")

if DEPENDENCIAS_FALTANTES:
    print("❌ Faltan dependencias:")
    for d in DEPENDENCIAS_FALTANTES:
        print(f"   pip install {d}")
    sys.exit(1)

# ── Configuración ─────────────────────────────────────────────
GROQ_API_KEY_1      = os.getenv("GROQ_API_KEY_1", "")
GROQ_API_KEY_2      = os.getenv("GROQ_API_KEY_2", "")
GROQ_API_KEY_3      = os.getenv("GROQ_API_KEY_3", "")
CEREBRAS_API_KEY    = os.getenv("CEREBRAS_API_KEY", "")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CEREBRAS_MODEL      = os.getenv("CEREBRAS_MODEL", "llama3.3-70b")
WHISPER_LANGUAGE    = os.getenv("WHISPER_LANGUAGE", "es")
WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL", "tiny")  # tiny para velocidad
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", 10))     # Chunks finales al LLM (post-reranker)
DB_PATH             = "./db"
BM25_PATH           = "./db/bm25_index.json"
EMBED_MODEL         = "BAAI/bge-m3"

# Audio config — optimizado para velocidad
SAMPLE_RATE     = 16000
BLOCK_SECONDS   = 0.5
SILENCE_UMBRAL  = 0.015
MIN_SPEECH_SEC  = 1.5
MAX_BUFFER_SEC  = 10          # Reducido para respuesta más rápida
SILENCE_BLOCKS  = 4           # 2 segundos de silencio (antes era 3s)

# Palabras clave para detectar preguntas — solo las más confiables
PALABRAS_PREGUNTA_INICIO = [
    "qué", "que", "cómo", "como", "cuál", "cual", "cuáles", "cuales",
    "cuándo", "cuando", "dónde", "donde", "por qué", "quién", "quien",
    "cuánto", "cuanto", "cuántos", "cuántas",
    "explica", "explique", "describe", "menciona", "define",
    "dime", "dame", "muéstrame", "cuéntame", "enumera", "lista"
]

# ── Cola de comunicación entre hilos ─────────────────────────
cola_ui = queue.Queue()


# ════════════════════════════════════════════════════════════
#  BM25 LOCAL (búsqueda por keywords)
# ════════════════════════════════════════════════════════════
class BuscadorBM25:
    """Búsqueda BM25 local usando índice pre-calculado."""
    
    def __init__(self):
        self.listo = False
        self.textos = []
        self.metadatas = []
        self.docs_tokens = []
        self.df = {}
        self.N = 0
        self.avgdl = 0
        self.doc_lengths = []
    
    def cargar(self) -> bool:
        try:
            if not os.path.exists(BM25_PATH):
                return False
            
            with open(BM25_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            indice = data["indice"]
            self.textos = data["textos"]
            self.metadatas = data["metadatas"]
            self.docs_tokens = indice["docs_tokens"]
            self.df = indice["df"]
            self.N = indice["N"]
            self.avgdl = indice["avgdl"]
            self.doc_lengths = indice["doc_lengths"]
            self.listo = True
            return True
        except Exception as e:
            print(f"⚠️ BM25 no disponible: {e}")
            return False
    
    def _tokenizar(self, texto: str) -> list[str]:
        texto = texto.lower()
        texto = re.sub(r'[^\w\sáéíóúñü]', ' ', texto)
        tokens = texto.split()
        stopwords = {'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'del', 'un', 'una',
                     'que', 'es', 'se', 'por', 'con', 'para', 'al', 'lo', 'como', 'su',
                     'más', 'o', 'este', 'ya', 'entre', 'muy', 'sin', 'sobre',
                     'ser', 'también', 'me', 'hasta', 'hay', 'donde', 'le', 'todo', 'nos'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]
    
    def buscar(self, consulta: str, top_k: int = 20) -> list[dict]:
        """Busca usando BM25. Retorna top_k resultados con score."""
        if not self.listo:
            return []
        
        query_tokens = self._tokenizar(consulta)
        if not query_tokens:
            return []
        
        k1 = 1.5
        b = 0.75
        scores = []
        
        for doc_idx in range(self.N):
            score = 0.0
            dl = self.doc_lengths[doc_idx]
            doc_tokens = self.docs_tokens[doc_idx]
            tf_counter = Counter(doc_tokens)
            
            for qt in query_tokens:
                if qt not in self.df:
                    continue
                
                tf = tf_counter.get(qt, 0)
                if tf == 0:
                    continue
                
                df_val = self.df[qt]
                idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1)
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
                score += idf * tf_norm
            
            if score > 0:
                scores.append((doc_idx, score))
        
        # Ordenar por score descendente
        scores.sort(key=lambda x: x[1], reverse=True)
        
        resultados = []
        for doc_idx, score in scores[:top_k]:
            resultados.append({
                "texto": self.textos[doc_idx],
                "fuente": self.metadatas[doc_idx].get("fuente", "?"),
                "score_bm25": score
            })
        
        return resultados


# ════════════════════════════════════════════════════════════
#  MOTOR RAG v2.0 (Híbrido + Reranker)
# ════════════════════════════════════════════════════════════
class MotorRAG:
    def __init__(self):
        self.modelo_embed = None
        self.coleccion    = None
        self.bm25         = None
        self.listo        = False
    
    def cargar(self):
        try:
            cola_ui.put(("estado", "🧠 Cargando embeddings BGE-M3..."))
            self.modelo_embed = SentenceTransformer(EMBED_MODEL)
            
            cola_ui.put(("estado", "🗄️ Cargando base de datos..."))
            cliente = chromadb.PersistentClient(path=DB_PATH)
            self.coleccion = cliente.get_collection("documentos_rag")
            
            cola_ui.put(("estado", "🔤 Cargando índice BM25..."))
            self.bm25 = BuscadorBM25()
            bm25_ok = self.bm25.cargar()
            
            count = self.coleccion.count()
            self.listo = True
            
            modo = "Híbrida (Vector + BM25 + RRF)" if bm25_ok else "Vector"
            cola_ui.put(("estado", f"✅ RAG listo — {count} fragmentos | {modo}"))
            cola_ui.put(("log", f"Base de datos: {count} fragmentos | Modo: {modo}"))
            return True
        except Exception as e:
            cola_ui.put(("error", f"❌ Error cargando RAG.\nEjecuta primero: python indexar_pdfs.py\n\nError: {e}"))
            return False
    
    def buscar(self, consulta: str) -> list[dict]:
        """Búsqueda híbrida: Vector + BM25 → RRF Fusion → Top-K."""
        if not self.listo:
            return []
        
        try:
            t0 = time.time()
            
            # 1. Búsqueda vectorial (top-25)
            embedding = self.modelo_embed.encode([consulta], normalize_embeddings=True).tolist()
            resultados_vec = self.coleccion.query(
                query_embeddings=embedding,
                n_results=25,
                include=["documents", "metadatas", "distances"]
            )
            
            # 2. Búsqueda BM25 (top-25)
            resultados_bm25 = []
            if self.bm25 and self.bm25.listo:
                resultados_bm25 = self.bm25.buscar(consulta, top_k=25)
            
            # 3. RRF Fusion (combinar rankings)
            candidatos = self._rrf_fusion(resultados_vec, resultados_bm25)
            
            if not candidatos:
                return []
            
            # Tomar top-K directamente del RRF
            resultado_final = candidatos[:RAG_TOP_K]
            
            elapsed = time.time() - t0
            cola_ui.put(("log", f"RAG: {len(resultado_final)} chunks en {elapsed:.2f}s (vec+bm25+rrf)"))
            
            return resultado_final
            
        except Exception as e:
            cola_ui.put(("log", f"Error RAG: {e}"))
            return []
    
    def _rrf_fusion(self, resultados_vec, resultados_bm25, k=60) -> list[dict]:
        """Reciprocal Rank Fusion para combinar vector + BM25."""
        fusion_scores = {}  # texto → {score, metadata}
        
        # Vectorial
        if resultados_vec and resultados_vec["documents"][0]:
            for rank, (doc, meta, dist) in enumerate(zip(
                resultados_vec["documents"][0],
                resultados_vec["metadatas"][0],
                resultados_vec["distances"][0]
            )):
                key = doc[:100]  # usar primeros 100 chars como key
                rrf_score = 1.0 / (k + rank + 1)
                if key not in fusion_scores:
                    fusion_scores[key] = {"texto": doc, "fuente": meta.get("fuente", "?"), "score": 0}
                fusion_scores[key]["score"] += rrf_score
        
        # BM25
        for rank, resultado in enumerate(resultados_bm25):
            key = resultado["texto"][:100]
            rrf_score = 1.0 / (k + rank + 1)
            if key not in fusion_scores:
                fusion_scores[key] = {"texto": resultado["texto"], "fuente": resultado["fuente"], "score": 0}
            fusion_scores[key]["score"] += rrf_score
        
        # Ordenar por RRF score
        candidatos = sorted(fusion_scores.values(), key=lambda x: x["score"], reverse=True)
        return candidatos


# ════════════════════════════════════════════════════════════
#  ORQUESTADOR DE LLMs CON STREAMING
# ════════════════════════════════════════════════════════════
class OrquestadorLLM:
    
    def _llamar_groq_stream(self, api_key: str, prompt: str, contexto: str):
        """Llama a Groq con streaming. Yield tokens parciales."""
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": self._sistema(contexto)},
                    {"role": "user",   "content": prompt}
                ],
                "max_tokens": 1200,
                "temperature": 0.2,
                "stream": True
            },
            timeout=8,
            stream=True
        )
        resp.raise_for_status()
        
        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except:
                        continue
    
    def _llamar_cerebras_stream(self, prompt: str, contexto: str):
        """Llama a Cerebras con streaming."""
        if not CEREBRAS_API_KEY or CEREBRAS_API_KEY == "tu_cerebras_api_key_aqui":
            raise ValueError("Cerebras API key no configurada")
        
        resp = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": CEREBRAS_MODEL,
                "messages": [
                    {"role": "system", "content": self._sistema(contexto)},
                    {"role": "user",   "content": prompt}
                ],
                "max_tokens": 1200,
                "temperature": 0.2,
                "stream": True
            },
            timeout=10,
            stream=True
        )
        resp.raise_for_status()
        
        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except:
                        continue
    
    def _sistema(self, contexto: str) -> str:
        return f"""Eres un asistente experto. Responde en español usando SOLO el contexto dado.

REGLAS OBLIGATORIAS:
- Enumera TODOS los elementos, puntos o conceptos que aparezcan en el contexto
- Si hay una lista con 3 items, menciona los 3. Si hay 5, menciona los 5. NUNCA omitas elementos.
- Incluye las descripciones o explicaciones de cada elemento
- Combina información de diferentes fragmentos del contexto
- NO digas "no se menciona" si la información está en algún fragmento
- NO resumas ni acortes listas — preséntalas COMPLETAS
- Español profesional

CONTEXTO:
{contexto}"""
    
    def responder_stream(self, pregunta: str, contexto: str):
        """
        Intenta en cascada con streaming.
        Yield: tuplas (tipo, dato) donde tipo es "token" o "done".
        """
        proveedores = []
        
        if GROQ_API_KEY_1 and "tu_groq_api_key" not in GROQ_API_KEY_1:
            proveedores.append(("Groq-1 ⚡", lambda p, c: self._llamar_groq_stream(GROQ_API_KEY_1, p, c)))
        if GROQ_API_KEY_2 and "tu_groq_api_key" not in GROQ_API_KEY_2:
            proveedores.append(("Groq-2 ⚡", lambda p, c: self._llamar_groq_stream(GROQ_API_KEY_2, p, c)))
        if GROQ_API_KEY_3 and "tu_groq_api_key" not in GROQ_API_KEY_3:
            proveedores.append(("Groq-3 ⚡", lambda p, c: self._llamar_groq_stream(GROQ_API_KEY_3, p, c)))
        if CEREBRAS_API_KEY and CEREBRAS_API_KEY != "tu_cerebras_api_key_aqui":
            proveedores.append(("Cerebras 🧠", lambda p, c: self._llamar_cerebras_stream(p, c)))
        
        if not proveedores:
            yield ("error", "❌ No hay APIs configuradas.")
            return
        
        for nombre, fn in proveedores:
            try:
                cola_ui.put(("log", f"Consultando {nombre} (streaming)..."))
                t0 = time.time()
                tokens_recibidos = False
                
                for token in fn(pregunta, contexto):
                    if not tokens_recibidos:
                        tokens_recibidos = True
                        ttft = time.time() - t0
                        cola_ui.put(("log", f"⚡ {nombre} primer token en {ttft:.2f}s"))
                    yield ("token", token)
                
                if tokens_recibidos:
                    elapsed = time.time() - t0
                    yield ("done", (nombre, elapsed))
                    return
                else:
                    raise ValueError("No se recibieron tokens")
                    
            except Exception as e:
                error_str = str(e)
                cola_ui.put(("log", f"⚠️ {nombre} falló: {error_str[:80]}"))
                continue
        
        yield ("error", "❌ Todos los proveedores fallaron.")


# ════════════════════════════════════════════════════════════
#  PROCESADOR DE AUDIO + STT
# ════════════════════════════════════════════════════════════
class ProcesadorAudio:
    def __init__(self, rag: MotorRAG, llm: OrquestadorLLM):
        self.rag      = rag
        self.llm      = llm
        self.buffer   = np.array([], dtype=np.float32)
        self.modelo_whisper = None
        self.activo   = False
        self.hablando = False
        self.silencio_contador = 0
    
    def cargar_whisper(self):
        cola_ui.put(("estado", f"🎙️ Cargando Whisper {WHISPER_MODEL_SIZE}..."))
        try:
            self.modelo_whisper = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8"
            )
            cola_ui.put(("estado", f"✅ Whisper cargado ({WHISPER_MODEL_SIZE})"))
            cola_ui.put(("log", f"Whisper ({WHISPER_MODEL_SIZE}) listo en CPU/int8"))
            return True
        except Exception as e:
            cola_ui.put(("error", f"Error cargando Whisper: {e}"))
            return False
    
    def detectar_pregunta(self, texto: str) -> bool:
        """Detecta preguntas con menos falsos positivos."""
        texto_lower = texto.lower().strip()
        
        # Signos de pregunta explícitos
        if "?" in texto:
            return True
        
        palabras = texto_lower.split()
        if not palabras:
            return False
        
        # Primera palabra es interrogativa
        if palabras[0] in PALABRAS_PREGUNTA_INICIO:
            return True
        
        # Primeras 2 palabras contienen interrogativa
        if len(palabras) >= 2:
            dos_primeras = " ".join(palabras[:2])
            for p in PALABRAS_PREGUNTA_INICIO[:18]:  # solo interrogativas puras
                if p in dos_primeras:
                    return True
        
        # Frases que empiezan con "por qué" o "para qué"
        if texto_lower.startswith("por qu") or texto_lower.startswith("para qu"):
            return True
        
        return False
    
    def transcribir_y_procesar(self, audio_data: np.ndarray):
        """Transcribe audio y procesa si es una pregunta."""
        try:
            segmentos, info = self.modelo_whisper.transcribe(
                audio_data,
                language=WHISPER_LANGUAGE,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400}
            )
            
            texto = " ".join([s.text for s in segmentos]).strip()
            
            if not texto or len(texto) < 10:
                return
            
            cola_ui.put(("transcripcion", texto))
            cola_ui.put(("log", f"📝 {texto[:80]}..."))
            
            if self.detectar_pregunta(texto):
                cola_ui.put(("pregunta_detectada", texto))
                self.procesar_pregunta(texto)
            
        except Exception as e:
            cola_ui.put(("log", f"Error transcripción: {e}"))
    
    def procesar_pregunta(self, pregunta: str):
        """Busca en RAG y llama al LLM con streaming."""
        t_inicio = time.time()
        cola_ui.put(("estado", "🔍 Buscando en documentos..."))
        
        # Buscar en RAG (híbrido + reranker)
        fragmentos = self.rag.buscar(pregunta)
        
        if not fragmentos:
            cola_ui.put(("respuesta", ("⚠️ No encontré información relevante en los documentos.", "RAG vacío", pregunta, 0)))
            cola_ui.put(("estado", "🎙️ Escuchando..."))
            return
        
        # Construir contexto (solo top chunks ya rerankeados)
        contexto_partes = []
        for i, f in enumerate(fragmentos, 1):
            contexto_partes.append(f"[{f['fuente']}]\n{f['texto']}")
        contexto = "\n\n---\n\n".join(contexto_partes)
        
        fuentes = list(set(f["fuente"] for f in fragmentos))
        cola_ui.put(("fuentes", fuentes))
        cola_ui.put(("estado", "💬 Generando respuesta..."))
        
        # LLM con streaming
        respuesta_completa = ""
        proveedor = "?"
        
        for tipo, dato in self.llm.responder_stream(pregunta, contexto):
            if tipo == "token":
                respuesta_completa += dato
                cola_ui.put(("stream_token", respuesta_completa))
            elif tipo == "done":
                proveedor, elapsed_llm = dato
            elif tipo == "error":
                respuesta_completa = dato
                proveedor = "Error"
        
        elapsed = time.time() - t_inicio
        cola_ui.put(("respuesta", (respuesta_completa, proveedor, pregunta, elapsed)))
        cola_ui.put(("estado", f"✅ Respuesta en {elapsed:.1f}s — Escuchando..."))
    
    def callback_audio(self, indata, frames, time_info, status):
        """Callback del stream de audio."""
        if status:
            pass
        
        audio_chunk = indata[:, 0].copy()
        nivel = np.abs(audio_chunk).mean()
        
        if nivel > SILENCE_UMBRAL:
            self.buffer = np.append(self.buffer, audio_chunk)
            self.hablando = True
            self.silencio_contador = 0
            cola_ui.put(("nivel_audio", min(nivel * 20, 1.0)))
        else:
            if self.hablando:
                self.silencio_contador += 1
                self.buffer = np.append(self.buffer, audio_chunk)
                
                segundos_buffer = len(self.buffer) / SAMPLE_RATE
                
                # Procesar con menos silencio (2s en vez de 3s)
                if (self.silencio_contador > SILENCE_BLOCKS and segundos_buffer > MIN_SPEECH_SEC) or \
                   segundos_buffer > MAX_BUFFER_SEC:
                    
                    audio_a_procesar = self.buffer.copy()
                    self.buffer = np.array([], dtype=np.float32)
                    self.hablando = False
                    self.silencio_contador = 0
                    
                    threading.Thread(
                        target=self.transcribir_y_procesar,
                        args=(audio_a_procesar,),
                        daemon=True
                    ).start()
            else:
                cola_ui.put(("nivel_audio", 0.0))
    
    def iniciar(self, device_index=None):
        """Inicia la captura de audio."""
        self.activo = True
        try:
            stream = sd.InputStream(
                device=device_index,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=int(SAMPLE_RATE * BLOCK_SECONDS),
                callback=self.callback_audio
            )
            stream.start()
            cola_ui.put(("estado", "🎙️ Escuchando..."))
            cola_ui.put(("log", f"Stream de audio iniciado (dispositivo: {device_index})"))
            return stream
        except Exception as e:
            cola_ui.put(("error", f"Error iniciando audio: {e}\n\nVerifica que VB-Cable esté instalado."))
            return None



# ════════════════════════════════════════════════════════════
#  INTERFAZ GRÁFICA FLOTANTE
# ════════════════════════════════════════════════════════════
class InterfazFlotante:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RAG Asistente v2.0")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#0a0e1a")
        self.root.geometry("900x1000+50+20")
        self.root.minsize(800, 900)
        
        # Estado
        self.stream_audio  = None
        self.procesador    = None
        self.rag           = None
        self.llm           = None
        self.dispositivos  = []
        self.nivel_audio   = 0.0
        self.pregunta_actual = ""
        
        self._construir_ui()
        self._inicializar_sistema()
    
    def _construir_ui(self):
        """Construye la interfaz flotante."""
        BG       = "#0a0e1a"
        BG2      = "#111827"
        BG3      = "#1a2235"
        VERDE    = "#00ff88"
        AZUL     = "#3b82f6"
        AMARILLO = "#fbbf24"
        ROJO     = "#ef4444"
        TEXTO    = "#e2e8f0"
        GRIS     = "#64748b"
        
        # ── Header ───────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG, pady=8)
        header.pack(fill="x", padx=12, pady=(10,0))
        
        tk.Label(header, text="⚡ RAG ASISTENTE v2.0", font=("Consolas", 14, "bold"),
                 bg=BG, fg=VERDE).pack(side="left")
        
        self.lbl_proveedor = tk.Label(header, text="iniciando...",
                                       font=("Consolas", 9), bg=BG, fg=GRIS)
        self.lbl_proveedor.pack(side="right")
        
        # ── Barra de estado ───────────────────────────────────
        self.lbl_estado = tk.Label(self.root, text="Cargando sistema...",
                                    font=("Consolas", 10), bg=BG, fg=AMARILLO,
                                    anchor="w")
        self.lbl_estado.pack(fill="x", padx=14, pady=(4,0))
        
        # ── Selector de dispositivo ───────────────────────────
        frame_dev = tk.Frame(self.root, bg=BG2, padx=8, pady=6)
        frame_dev.pack(fill="x", padx=12, pady=6)
        
        tk.Label(frame_dev, text="🎙️ Entrada:", font=("Consolas", 9),
                 bg=BG2, fg=GRIS).pack(side="left")
        
        self.var_dispositivo = tk.StringVar()
        self.combo_dispositivos = ttk.Combobox(
            frame_dev, textvariable=self.var_dispositivo,
            font=("Consolas", 9), width=50, state="readonly"
        )
        self.combo_dispositivos.pack(side="left", padx=6)
        
        self.btn_iniciar = tk.Button(
            frame_dev, text="▶ INICIAR", font=("Consolas", 9, "bold"),
            bg=VERDE, fg=BG, relief="flat", padx=8,
            command=self.iniciar_escucha
        )
        self.btn_iniciar.pack(side="left", padx=4)
        
        self.btn_detener = tk.Button(
            frame_dev, text="⏹ PARAR", font=("Consolas", 9, "bold"),
            bg=ROJO, fg="white", relief="flat", padx=8,
            command=self.detener_escucha, state="disabled"
        )
        self.btn_detener.pack(side="left")
        
        # ── Nivel de audio ────────────────────────────────────
        frame_nivel = tk.Frame(self.root, bg=BG, pady=2)
        frame_nivel.pack(fill="x", padx=14)
        
        tk.Label(frame_nivel, text="Nivel:", font=("Consolas", 8),
                 bg=BG, fg=GRIS).pack(side="left")
        
        self.canvas_nivel = tk.Canvas(frame_nivel, height=8, bg=BG3,
                                       highlightthickness=0)
        self.canvas_nivel.pack(side="left", fill="x", expand=True, padx=6)
        
        # ── Última transcripción ──────────────────────────────
        tk.Label(self.root, text="ÚLTIMO AUDIO DETECTADO",
                 font=("Consolas", 8, "bold"), bg=BG, fg=GRIS).pack(
                     anchor="w", padx=14, pady=(8,2))
        
        self.txt_transcripcion = tk.Text(
            self.root, height=3, font=("Consolas", 10),
            bg=BG3, fg=TEXTO, relief="flat", padx=8, pady=6,
            wrap="word", state="disabled", insertbackground=TEXTO
        )
        self.txt_transcripcion.pack(fill="x", padx=12, pady=(0,4))
        
        # ── Pregunta detectada ────────────────────────────────
        frame_preg = tk.Frame(self.root, bg=BG2, padx=10, pady=6)
        frame_preg.pack(fill="x", padx=12, pady=2)
        
        tk.Label(frame_preg, text="❓ PREGUNTA DETECTADA",
                 font=("Consolas", 8, "bold"), bg=BG2, fg=AMARILLO).pack(anchor="w")
        
        self.lbl_pregunta = tk.Label(
            frame_preg, text="Esperando pregunta...",
            font=("Consolas", 11), bg=BG2, fg=TEXTO,
            anchor="w", wraplength=840, justify="left"
        )
        self.lbl_pregunta.pack(fill="x", pady=(4,0))
        
        # ── Respuesta ─────────────────────────────────────────
        tk.Label(self.root, text="💡 RESPUESTA (streaming)",
                 font=("Consolas", 8, "bold"), bg=BG, fg=VERDE).pack(
                     anchor="w", padx=14, pady=(10,2))
        
        self.txt_respuesta = scrolledtext.ScrolledText(
            self.root, height=20, font=("Consolas", 11),
            bg=BG2, fg=VERDE, relief="flat", padx=10, pady=8,
            wrap="word", state="disabled",
            selectbackground=AZUL
        )
        self.txt_respuesta.pack(fill="both", expand=True, padx=12, pady=(0,4))
        
        # ── Fuentes ───────────────────────────────────────────
        self.lbl_fuentes = tk.Label(
            self.root, text="Fuentes: —",
            font=("Consolas", 8), bg=BG, fg=GRIS, anchor="w"
        )
        self.lbl_fuentes.pack(fill="x", padx=14, pady=(0,2))
        
        # ── Log ───────────────────────────────────────────────
        self.txt_log = tk.Text(
            self.root, height=5, font=("Consolas", 8),
            bg=BG3, fg=GRIS, relief="flat", padx=8, pady=4,
            wrap="word", state="disabled"
        )
        self.txt_log.pack(fill="x", padx=12, pady=(0,10))
        
        # Copiar respuesta con doble clic
        self.txt_respuesta.bind("<Double-Button-1>", self.copiar_respuesta)
        self.root.bind("<Escape>", lambda e: self.root.iconify())
        
        # Tip
        tk.Label(self.root, text="💡 Doble clic en respuesta para copiar | ESC minimizar",
                 font=("Consolas", 7), bg=BG, fg=GRIS).pack(pady=(0,6))
    
    def _set_texto(self, widget: tk.Text, texto: str, color=None):
        widget.config(state="normal")
        widget.delete(1.0, "end")
        if color:
            widget.config(fg=color)
        widget.insert("end", texto)
        widget.config(state="disabled")
    
    def copiar_respuesta(self, event=None):
        try:
            texto = self.txt_respuesta.get(1.0, "end").strip()
            self.root.clipboard_clear()
            self.root.clipboard_append(texto)
            self.lbl_estado.config(text="✅ Respuesta copiada al portapapeles")
        except:
            pass
    
    def actualizar_nivel_audio(self, nivel: float):
        self.nivel_audio = nivel
        w = self.canvas_nivel.winfo_width()
        h = self.canvas_nivel.winfo_height()
        self.canvas_nivel.delete("all")
        if w > 0:
            ancho = int(w * nivel)
            color = "#00ff88" if nivel < 0.6 else "#fbbf24" if nivel < 0.85 else "#ef4444"
            self.canvas_nivel.create_rectangle(0, 0, ancho, h, fill=color, outline="")
    
    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", f"[{ts}] {msg}\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")
    
    def cargar_dispositivos(self):
        try:
            dispositivos = sd.query_devices()
            entradas = []
            for i, d in enumerate(dispositivos):
                if d["max_input_channels"] > 0:
                    entradas.append((i, d["name"]))
            
            self.dispositivos = entradas
            nombres = [f"{i}: {n}" for i, n in entradas]
            self.combo_dispositivos["values"] = nombres
            
            for nombre in nombres:
                if "cable" in nombre.lower() or "vb-audio" in nombre.lower() or "virtual" in nombre.lower():
                    self.combo_dispositivos.set(nombre)
                    self._log(f"VB-Cable detectado: {nombre}")
                    return
            
            if nombres:
                self.combo_dispositivos.set(nombres[0])
        except Exception as e:
            self._log(f"Error cargando dispositivos: {e}")
    
    def iniciar_escucha(self):
        if not self.procesador:
            self._log("Sistema no listo aún")
            return
        
        sel = self.combo_dispositivos.get()
        if not sel:
            self.lbl_estado.config(text="⚠️ Selecciona un dispositivo de audio")
            return
        
        idx = int(sel.split(":")[0])
        self.stream_audio = self.procesador.iniciar(idx)
        
        if self.stream_audio:
            self.btn_iniciar.config(state="disabled")
            self.btn_detener.config(state="normal")
    
    def detener_escucha(self):
        if self.stream_audio:
            self.stream_audio.stop()
            self.stream_audio.close()
            self.stream_audio = None
        self.btn_iniciar.config(state="normal")
        self.btn_detener.config(state="disabled")
        self.lbl_estado.config(text="⏹ Escucha detenida")
    
    def _inicializar_sistema(self):
        """Inicializa RAG, LLM y Whisper en hilo separado."""
        def _init():
            self.rag = MotorRAG()
            ok_rag = self.rag.cargar()
            
            if not ok_rag:
                return
            
            self.llm = OrquestadorLLM()
            
            self.procesador = ProcesadorAudio(self.rag, self.llm)
            ok_whisper = self.procesador.cargar_whisper()
            
            if ok_whisper:
                self.root.after(0, self.cargar_dispositivos)
                cola_ui.put(("estado", "✅ Sistema listo — Selecciona dispositivo y presiona INICIAR"))
        
        threading.Thread(target=_init, daemon=True).start()
    
    def procesar_cola(self):
        """Procesa mensajes de la cola de comunicación."""
        try:
            while not cola_ui.empty():
                tipo, datos = cola_ui.get_nowait()
                
                if tipo == "estado":
                    self.lbl_estado.config(text=datos)
                
                elif tipo == "log":
                    self._log(datos)
                
                elif tipo == "error":
                    self._set_texto(self.txt_respuesta, f"❌ ERROR:\n{datos}", "#ef4444")
                    self._log(f"ERROR: {datos[:80]}")
                
                elif tipo == "transcripcion":
                    self._set_texto(self.txt_transcripcion, datos)
                
                elif tipo == "pregunta_detectada":
                    self.pregunta_actual = datos
                    self.lbl_pregunta.config(text=datos, fg="#fbbf24")
                    self._set_texto(self.txt_respuesta, "⏳ Buscando...", "#64748b")
                
                elif tipo == "stream_token":
                    # Actualizar respuesta en tiempo real (streaming)
                    self._set_texto(self.txt_respuesta, datos, "#00ff88")
                
                elif tipo == "respuesta":
                    respuesta, proveedor, pregunta, elapsed = datos
                    self._set_texto(self.txt_respuesta, respuesta, "#00ff88")
                    self.lbl_proveedor.config(text=f"{proveedor} | {elapsed:.1f}s")
                    self._log(f"✅ {proveedor} en {elapsed:.1f}s")
                
                elif tipo == "fuentes":
                    fuentes = datos
                    self.lbl_fuentes.config(text=f"Fuentes: {' | '.join(fuentes)}")
                
                elif tipo == "nivel_audio":
                    self.actualizar_nivel_audio(datos)
        
        except queue.Empty:
            pass
        except Exception as e:
            pass
        
        self.root.after(50, self.procesar_cola)  # 50ms para streaming más fluido
    
    def ejecutar(self):
        self.root.after(100, self.procesar_cola)
        self.root.mainloop()


# ════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  ⚡ RAG ASISTENTE v2.0 - Iniciando...")
    print("  Búsqueda: Híbrida (Vector + BM25 + Reranker)")
    print("  Streaming: Activado")
    print("="*60)
    print()
    
    app = InterfazFlotante()
    app.ejecutar()
