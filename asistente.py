"""
================================================================
RAG ASISTENTE v2.0 - ESCUCHA REUNIONES Y RESPONDE CON TUS DOCUMENTOS
================================================================
Arquitectura:
  Teams Audio → VB-Cable → faster-whisper → Búsqueda Híbrida (Vector + BM25)
  → RRF Fusion → Top-10 chunks → LLM Streaming → UI Flotante

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
import ctypes
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

# Audio config — balanceado para capturar preguntas completas
SAMPLE_RATE     = 16000
BLOCK_SECONDS   = 0.3           # Bloques de 300ms
SILENCE_UMBRAL  = 0.008         # Más sensible para no perder audio suave
MIN_SPEECH_SEC  = 1.5           # Mínimo para procesar
MAX_BUFFER_SEC  = 25            # Permite preguntas largas (~25 seg)
SILENCE_BLOCKS  = 10            # 10 × 0.3s = 3.0s de silencio para cortar

# Palabras clave para detectar preguntas — solo las más confiables
PALABRAS_PREGUNTA_INICIO = [
    "qué", "que", "cómo", "como", "cuál", "cual", "cuáles", "cuales",
    "cuándo", "cuando", "dónde", "donde", "por qué", "quién", "quien",
    "cuánto", "cuanto", "cuántos", "cuántas",
    "explica", "explique", "describe", "menciona", "define",
    "dime", "dame", "muéstrame", "cuéntame", "enumera", "lista",
    "háblame", "hablame", "habla", "háblanos", "hablanos"
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
        """Detecta preguntas de forma tolerante (Whisper puede perder la primera palabra)."""
        texto_lower = texto.lower().strip()
        
        # Signos de pregunta explícitos
        if "?" in texto:
            return True
        
        palabras = texto_lower.split()
        if not palabras:
            return False
        
        # Palabras interrogativas — buscar en las primeras 5 palabras
        # (Whisper a veces corta "cuál" y queda "es el límite...")
        interrogativas = [
            "qué", "que", "cómo", "como", "cuál", "cual", "cuáles", "cuales",
            "cuándo", "cuando", "dónde", "donde", "por qué", "quién", "quien",
            "cuánto", "cuanto", "cuántos", "cuántas"
        ]
        
        # Comandos directos — buscar en las primeras 3 palabras
        comandos = [
            "explica", "explique", "describe", "menciona", "define",
            "dime", "dame", "muéstrame", "cuéntame", "enumera", "lista",
            "háblame", "hablame", "habla", "háblanos", "hablanos"
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
        
        # Frases parciales (por si Whisper corta el inicio)
        texto_inicio = " ".join(palabras[:6])
        if "por qu" in texto_inicio or "para qu" in texto_inicio:
            return True
        
        # Patrones comunes que indican pregunta aunque falte la primera palabra
        # ej: "es el límite máximo..." (faltó "cuál")
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
        # Activar DPI awareness ANTES de crear la ventana
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI aware
        except:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except:
                pass
        
        self.root = tk.Tk()
        self.root.title("RAG Asistente v2.0")
        self.root.configure(bg="#0f1219")
        # Tamaño fijo que NO cubre la barra de tareas (deja 60px abajo)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_h = screen_h - 80  # Dejar espacio para barra de tareas
        self.root.geometry(f"960x{win_h}+50+10")
        self.root.minsize(850, 600)
        
        # Siempre encima pero permite interactuar con barra de tareas
        self.root.attributes("-topmost", True)
        
        # Al perder foco, bajar prioridad temporalmente para acceder a otras apps
        self.root.bind("<FocusOut>", self._on_focus_out)
        self.root.bind("<FocusIn>", self._on_focus_in)
        
        # Escalar DPI
        self.dpi_scale = self.root.winfo_fpixels('1i') / 96.0
        
        # Estado
        self.stream_audio  = None
        self.procesador    = None
        self.rag           = None
        self.llm           = None
        self.dispositivos  = []
        self.nivel_audio   = 0.0
        self.pregunta_actual = ""
        
        # Estilo ttk
        self._configurar_estilo()
        self._construir_ui()
        self._inicializar_sistema()
    
    def _configurar_estilo(self):
        """Configura estilos ttk para un look más profesional."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Combobox
        style.configure('Custom.TCombobox',
            fieldbackground='#1e2736',
            background='#2a3a4e',
            foreground='#e2e8f0',
            borderwidth=0,
            relief='flat'
        )
        style.map('Custom.TCombobox',
            fieldbackground=[('readonly', '#1e2736')],
            foreground=[('readonly', '#e2e8f0')]
        )
    
    def _construir_ui(self):
        """Construye la interfaz flotante con diseño profesional."""
        # Paleta de colores
        BG       = "#0f1219"   # Fondo principal (más oscuro)
        BG2      = "#161d2b"   # Paneles
        BG3      = "#1e2736"   # Inputs/campos
        ACCENT   = "#10b981"   # Verde esmeralda (más profesional)
        AZUL     = "#3b82f6"
        AMARILLO = "#f59e0b"
        ROJO     = "#ef4444"
        TEXTO    = "#f1f5f9"
        GRIS     = "#94a3b8"
        BORDE    = "#2a3a4e"
        
        # Fuentes
        FONT_TITLE  = ("Segoe UI", 13, "bold")
        FONT_BODY   = ("Segoe UI", 10)
        FONT_SMALL  = ("Segoe UI", 9)
        FONT_MONO   = ("Cascadia Code", 10)
        FONT_LOG    = ("Cascadia Code", 8)
        FONT_LABEL  = ("Segoe UI", 8, "bold")
        
        # ── Header ───────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG, pady=10)
        header.pack(fill="x", padx=16, pady=(12,0))
        
        tk.Label(header, text="⚡ RAG ASISTENTE", font=FONT_TITLE,
                 bg=BG, fg=ACCENT).pack(side="left")
        
        self.lbl_proveedor = tk.Label(header, text="iniciando...",
                                       font=FONT_SMALL, bg=BG, fg=GRIS)
        self.lbl_proveedor.pack(side="right")
        
        # Separador
        tk.Frame(self.root, bg=BORDE, height=1).pack(fill="x", padx=16, pady=(8,0))
        
        # ── Barra de estado ───────────────────────────────────
        self.lbl_estado = tk.Label(self.root, text="⏳ Cargando sistema...",
                                    font=FONT_BODY, bg=BG, fg=AMARILLO,
                                    anchor="w")
        self.lbl_estado.pack(fill="x", padx=18, pady=(8,0))
        
        # ── Selector de dispositivo ───────────────────────────
        frame_dev = tk.Frame(self.root, bg=BG2, padx=12, pady=10)
        frame_dev.pack(fill="x", padx=16, pady=(10,0))
        
        tk.Label(frame_dev, text="🎙️ Entrada de audio:", font=FONT_SMALL,
                 bg=BG2, fg=GRIS).pack(side="left")
        
        self.var_dispositivo = tk.StringVar()
        self.combo_dispositivos = ttk.Combobox(
            frame_dev, textvariable=self.var_dispositivo,
            font=FONT_SMALL, width=45, state="readonly",
            style='Custom.TCombobox'
        )
        self.combo_dispositivos.pack(side="left", padx=8)
        
        self.btn_iniciar = tk.Button(
            frame_dev, text="▶ INICIAR", font=("Segoe UI", 9, "bold"),
            bg=ACCENT, fg="#000000", relief="flat", padx=12, pady=4,
            activebackground="#059669", cursor="hand2",
            command=self.iniciar_escucha
        )
        self.btn_iniciar.pack(side="left", padx=4)
        
        self.btn_detener = tk.Button(
            frame_dev, text="⏹ PARAR", font=("Segoe UI", 9, "bold"),
            bg=ROJO, fg="white", relief="flat", padx=12, pady=4,
            activebackground="#dc2626", cursor="hand2",
            command=self.detener_escucha, state="disabled"
        )
        self.btn_detener.pack(side="left", padx=2)
        
        # ── Nivel de audio ────────────────────────────────────
        frame_nivel = tk.Frame(self.root, bg=BG, pady=4)
        frame_nivel.pack(fill="x", padx=18)
        
        tk.Label(frame_nivel, text="NIVEL", font=FONT_LABEL,
                 bg=BG, fg=GRIS).pack(side="left")
        
        self.canvas_nivel = tk.Canvas(frame_nivel, height=6, bg=BG3,
                                       highlightthickness=0, bd=0)
        self.canvas_nivel.pack(side="left", fill="x", expand=True, padx=(8,0))
        
        # ── Última transcripción ──────────────────────────────
        tk.Label(self.root, text="TRANSCRIPCIÓN",
                 font=FONT_LABEL, bg=BG, fg=GRIS).pack(
                     anchor="w", padx=18, pady=(10,3))
        
        self.txt_transcripcion = tk.Text(
            self.root, height=2, font=FONT_MONO,
            bg=BG3, fg=TEXTO, relief="flat", padx=10, pady=8,
            wrap="word", state="disabled", insertbackground=TEXTO,
            highlightthickness=1, highlightbackground=BORDE
        )
        self.txt_transcripcion.pack(fill="x", padx=16, pady=(0,4))
        
        # ── Pregunta detectada ────────────────────────────────
        frame_preg = tk.Frame(self.root, bg=BG2, padx=12, pady=8)
        frame_preg.pack(fill="x", padx=16, pady=4)
        
        tk.Label(frame_preg, text="❓ PREGUNTA DETECTADA",
                 font=FONT_LABEL, bg=BG2, fg=AMARILLO).pack(anchor="w")
        
        self.lbl_pregunta = tk.Label(
            frame_preg, text="Esperando pregunta...",
            font=("Segoe UI", 11), bg=BG2, fg=TEXTO,
            anchor="w", wraplength=880, justify="left"
        )
        self.lbl_pregunta.pack(fill="x", pady=(4,0))
        
        # ── Respuesta ─────────────────────────────────────────
        tk.Label(self.root, text="💡 RESPUESTA",
                 font=FONT_LABEL, bg=BG, fg=ACCENT).pack(
                     anchor="w", padx=18, pady=(10,3))
        
        self.txt_respuesta = scrolledtext.ScrolledText(
            self.root, height=18, font=FONT_MONO,
            bg=BG2, fg=ACCENT, relief="flat", padx=12, pady=10,
            wrap="word", state="disabled",
            selectbackground=AZUL, selectforeground="white",
            highlightthickness=1, highlightbackground=BORDE
        )
        self.txt_respuesta.pack(fill="both", expand=True, padx=16, pady=(0,4))
        
        # ── Fuentes ───────────────────────────────────────────
        self.lbl_fuentes = tk.Label(
            self.root, text="Fuentes: —",
            font=FONT_SMALL, bg=BG, fg=GRIS, anchor="w"
        )
        self.lbl_fuentes.pack(fill="x", padx=18, pady=(0,4))
        
        # ── Log ───────────────────────────────────────────────
        self.txt_log = tk.Text(
            self.root, height=4, font=FONT_LOG,
            bg=BG3, fg=GRIS, relief="flat", padx=10, pady=6,
            wrap="word", state="disabled",
            highlightthickness=1, highlightbackground=BORDE
        )
        self.txt_log.pack(fill="x", padx=16, pady=(0,8))
        
        # Copiar respuesta con doble clic
        self.txt_respuesta.bind("<Double-Button-1>", self.copiar_respuesta)
        self.root.bind("<Escape>", lambda e: self.root.iconify())
        
        # Footer
        tk.Label(self.root, text="Doble clic en respuesta para copiar  •  ESC minimizar",
                 font=("Segoe UI", 8), bg=BG, fg="#475569").pack(pady=(0,8))
    
    def _set_texto(self, widget: tk.Text, texto: str, color=None):
        widget.config(state="normal")
        widget.delete(1.0, "end")
        if color:
            widget.config(fg=color)
        widget.insert("end", texto)
        widget.config(state="disabled")
    
    def _on_focus_out(self, event=None):
        """Al perder foco, dejar de estar encima para acceder a otras apps."""
        self.root.attributes("-topmost", False)
    
    def _on_focus_in(self, event=None):
        """Al recuperar foco, volver a estar encima."""
        self.root.attributes("-topmost", True)
    
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
