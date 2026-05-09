"""
================================================================
RAG ASISTENTE - ESCUCHA REUNIONES Y RESPONDE CON TUS DOCUMENTOS
================================================================
Arquitectura:
  Teams Audio → VB-Cable → faster-whisper → ChromaDB RAG → Groq/OpenRouter/Gemini → UI Flotante

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
import traceback
from datetime import datetime
from pathlib import Path
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
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENROUTER_MODEL    = os.getenv("OPENROUTER_MODEL", "openrouter/free")
GEMINI_MODEL        = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
WHISPER_LANGUAGE    = os.getenv("WHISPER_LANGUAGE", "es")
WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL", "base")
RAG_TOP_K           = int(os.getenv("RAG_TOP_K", 12))  # Más contexto para capturar información completa
DB_PATH             = "./db"
EMBED_MODEL         = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Audio config
SAMPLE_RATE     = 16000
BLOCK_SECONDS   = 0.5
SILENCE_UMBRAL  = 0.015       # sensibilidad al silencio
MIN_SPEECH_SEC  = 1.5         # mínimo para procesar
MAX_BUFFER_SEC  = 12          # máximo antes de procesar

# Palabras clave para detectar preguntas (además de signos)
PALABRAS_PREGUNTA = [
    "qué", "que", "cómo", "como", "cuál", "cual", "cuáles", "cuales",
    "cuándo", "cuando", "dónde", "donde", "por qué", "porque", "quién",
    "quien", "cuánto", "cuanto", "explica", "explique", "describe",
    "menciona", "define", "definir", "nombra", "indica", "di",
    "habla", "comenta", "analiza", "compara", "relaciona",
    "puede", "podría", "alguien", "sabe", "conoce",
    "dime", "dame", "muestra", "muéstrame", "diga", "cuéntame",
    "enumera", "lista", "detalla", "especifica", "señala"
]

# ── Cola de comunicación entre hilos ─────────────────────────
cola_ui = queue.Queue()


# ════════════════════════════════════════════════════════════
#  MOTOR RAG
# ════════════════════════════════════════════════════════════
class MotorRAG:
    def __init__(self):
        self.modelo_embed = None
        self.coleccion    = None
        self.listo        = False
    
    def cargar(self):
        try:
            cola_ui.put(("estado", "🧠 Cargando embeddings..."))
            self.modelo_embed = SentenceTransformer(EMBED_MODEL)
            
            cola_ui.put(("estado", "🗄️ Cargando base de datos..."))
            cliente = chromadb.PersistentClient(path=DB_PATH)
            self.coleccion = cliente.get_collection("documentos_rag")
            
            count = self.coleccion.count()
            self.listo = True
            cola_ui.put(("estado", f"✅ RAG listo — {count} fragmentos indexados"))
            cola_ui.put(("log", f"Base de datos cargada: {count} fragmentos"))
            return True
        except Exception as e:
            cola_ui.put(("error", f"❌ Base de datos no encontrada.\nEjecuta primero: python indexar_pdfs.py\n\nError: {e}"))
            return False
    
    def buscar(self, consulta: str) -> list[dict]:
        if not self.listo:
            return []
        try:
            embedding = self.modelo_embed.encode([consulta]).tolist()
            resultados = self.coleccion.query(
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
                relevancia = 1 - dist  # cosine → similaridad
                if relevancia > 0.20:  # Umbral más bajo para capturar más contexto
                    fragmentos.append({
                        "texto": doc,
                        "fuente": meta.get("fuente", "?"),
                        "relevancia": relevancia
                    })
            
            return fragmentos
        except Exception as e:
            cola_ui.put(("log", f"Error RAG: {e}"))
            return []


# ════════════════════════════════════════════════════════════
#  ORQUESTADOR DE LLMs (Groq → OpenRouter → Gemini)
# ════════════════════════════════════════════════════════════
class OrquestadorLLM:
    
    def llamar_groq(self, prompt: str, contexto: str) -> str:
        if not GROQ_API_KEY or GROQ_API_KEY == "tu_groq_api_key_aqui":
            raise ValueError("Groq API key no configurada")
        
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": self._sistema(contexto)},
                    {"role": "user",   "content": prompt}
                ],
                "max_tokens": 1500,  # Respuestas más largas y detalladas
                "temperature": 0.3
            },
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    
    def llamar_openrouter(self, prompt: str, contexto: str) -> str:
        if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "tu_openrouter_api_key_aqui":
            raise ValueError("OpenRouter API key no configurada")
        
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "rag-asistente-local"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": self._sistema(contexto)},
                    {"role": "user",   "content": prompt}
                ],
                "max_tokens": 1500  # Respuestas más largas y detalladas
            },
            timeout=15
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    
    def llamar_gemini(self, prompt: str, contexto: str) -> str:
        if not GEMINI_API_KEY or GEMINI_API_KEY == "tu_gemini_api_key_aqui":
            raise ValueError("Gemini API key no configurada")
        
        mensaje = f"{self._sistema(contexto)}\n\nPregunta: {prompt}"
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
            json={"contents": [{"parts": [{"text": mensaje}]}]},
            timeout=15
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    def _sistema(self, contexto: str) -> str:
        return f"""Eres un asistente experto que proporciona respuestas completas y detalladas en español.

REGLAS CRÍTICAS:
- Proporciona respuestas COMPLETAS usando TODA la información disponible en el contexto
- Si el contexto contiene información fragmentada, UNELA y preséntala de forma coherente
- Si hay listas o múltiples elementos en CUALQUIER parte del contexto, enumera TODOS con sus descripciones completas
- Usa formato claro con numeración, viñetas o párrafos según sea apropiado
- NO digas "no está disponible" si la información existe en alguna parte del contexto
- Si encuentras información parcial en diferentes fragmentos, COMBÍNALA en una respuesta completa
- Explica conceptos de forma clara y profesional
- Usa solo la información del contexto proporcionado, pero úsala TODA
- Español neutro y profesional

IMPORTANTE: Si ves títulos como "Objetivo general", "Objetivos específicos", "Principios", etc., 
busca el contenido correspondiente en TODO el contexto y preséntalo completo.

CONTEXTO DE LOS DOCUMENTOS:
{contexto}"""
    
    def responder(self, pregunta: str, contexto: str) -> tuple[str, str]:
        """Intenta en orden: Groq → OpenRouter → Gemini. Retorna (respuesta, proveedor)"""
        
        proveedores = [
            ("Groq ⚡",       self.llamar_groq),
            ("OpenRouter 🔄", self.llamar_openrouter),
            ("Gemini 🌟",     self.llamar_gemini),
        ]
        
        ultimo_error = ""
        for nombre, fn in proveedores:
            try:
                cola_ui.put(("log", f"Consultando {nombre}..."))
                t0 = time.time()
                respuesta = fn(pregunta, contexto)
                elapsed = time.time() - t0
                cola_ui.put(("log", f"✅ {nombre} respondió en {elapsed:.1f}s"))
                return respuesta, nombre
            except Exception as e:
                ultimo_error = str(e)
                cola_ui.put(("log", f"⚠️ {nombre} falló: {e}"))
                continue
        
        return f"❌ Todos los proveedores fallaron. Último error: {ultimo_error}", "Error"


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
        self.dispositivo = None
        self.ultimo_proceso = time.time()
        self.hablando = False
        self.silencio_contador = 0
    
    def cargar_whisper(self):
        cola_ui.put(("estado", f"🎙️ Cargando Whisper {WHISPER_MODEL_SIZE}..."))
        try:
            self.modelo_whisper = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8"  # Optimizado para CPU
            )
            cola_ui.put(("estado", f"✅ Whisper cargado"))
            cola_ui.put(("log", f"Whisper ({WHISPER_MODEL_SIZE}) listo en CPU/int8"))
            return True
        except Exception as e:
            cola_ui.put(("error", f"Error cargando Whisper: {e}"))
            return False
    
    def detectar_pregunta(self, texto: str) -> bool:
        """Detecta si el texto contiene una pregunta."""
        texto_lower = texto.lower().strip()
        
        # Signos de pregunta
        if "?" in texto:
            return True
        
        # Palabras clave al inicio (más común)
        palabras = texto_lower.split()
        if palabras and palabras[0] in PALABRAS_PREGUNTA:
            return True
        
        # Cualquier palabra clave en las primeras 3 palabras
        if len(palabras) >= 2:
            primeras_palabras = " ".join(palabras[:3])
            for palabra in PALABRAS_PREGUNTA:
                if palabra in primeras_palabras:
                    return True
        
        # Cualquier palabra clave en el texto (si tiene más de 4 palabras)
        if len(palabras) > 4:
            for palabra in PALABRAS_PREGUNTA[:25]:  # las más importantes
                if palabra in texto_lower:
                    return True
        
        return False
    
    def transcribir_y_procesar(self, audio_data: np.ndarray):
        """Transcribe audio y procesa si es una pregunta."""
        try:
            # Transcribir
            segmentos, info = self.modelo_whisper.transcribe(
                audio_data,
                language=WHISPER_LANGUAGE,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500}
            )
            
            texto = " ".join([s.text for s in segmentos]).strip()
            
            if not texto or len(texto) < 10:
                return
            
            cola_ui.put(("transcripcion", texto))
            cola_ui.put(("log", f"Transcrito: {texto[:80]}..."))
            
            # ¿Es una pregunta?
            if self.detectar_pregunta(texto):
                cola_ui.put(("pregunta_detectada", texto))
                self.procesar_pregunta(texto)
            
        except Exception as e:
            cola_ui.put(("log", f"Error transcripción: {e}"))
    
    def procesar_pregunta(self, pregunta: str):
        """Busca en RAG y llama al LLM."""
        t_inicio = time.time()
        cola_ui.put(("estado", "🔍 Buscando en documentos..."))
        
        # Buscar en RAG
        fragmentos = self.rag.buscar(pregunta)
        
        if not fragmentos:
            cola_ui.put(("respuesta", ("⚠️ No encontré información relevante en los documentos.", "RAG vacío", pregunta, 0)))
            cola_ui.put(("estado", "🎙️ Escuchando..."))
            return
        
        # Construir contexto
        contexto_partes = []
        for i, f in enumerate(fragmentos, 1):
            contexto_partes.append(
                f"[Fuente: {f['fuente']} | Relevancia: {f['relevancia']:.0%}]\n{f['texto']}"
            )
        contexto = "\n\n---\n\n".join(contexto_partes)
        
        cola_ui.put(("log", f"RAG: {len(fragmentos)} fragmentos relevantes"))
        cola_ui.put(("estado", "💬 Consultando IA..."))
        
        # Llamar LLM
        respuesta, proveedor = self.llm.responder(pregunta, contexto)
        
        elapsed = time.time() - t_inicio
        fuentes = list(set(f["fuente"] for f in fragmentos))
        
        cola_ui.put(("respuesta", (respuesta, proveedor, pregunta, elapsed)))
        cola_ui.put(("fuentes", fuentes))
        cola_ui.put(("estado", f"✅ Respuesta en {elapsed:.1f}s — Escuchando..."))
    
    def callback_audio(self, indata, frames, time_info, status):
        """Callback del stream de audio."""
        if status:
            pass  # ignorar warnings menores
        
        audio_chunk = indata[:, 0].copy()
        nivel = np.abs(audio_chunk).mean()
        
        # Detectar si hay voz
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
                
                # Procesar si hay suficiente silencio o buffer muy grande
                if (self.silencio_contador > 6 and segundos_buffer > MIN_SPEECH_SEC) or \
                   segundos_buffer > MAX_BUFFER_SEC:
                    
                    audio_a_procesar = self.buffer.copy()
                    self.buffer = np.array([], dtype=np.float32)
                    self.hablando = False
                    self.silencio_contador = 0
                    
                    # Procesar en hilo separado para no bloquear audio
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
            cola_ui.put(("error", f"Error iniciando audio: {e}\n\nVerifica que VB-Cable esté instalado y seleccionado."))
            return None


# ════════════════════════════════════════════════════════════
#  INTERFAZ GRÁFICA FLOTANTE
# ════════════════════════════════════════════════════════════
class InterfazFlotante:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RAG Asistente")
        self.root.attributes("-topmost", True)  # Siempre encima
        # self.root.attributes("-alpha", 0.95)  # Transparencia desactivada
        self.root.configure(bg="#0a0e1a")
        # Ventana más grande para 2 pantallas
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
        
        tk.Label(header, text="⚡ RAG ASISTENTE", font=("Consolas", 14, "bold"),
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
        
        tk.Label(frame_dev, text="🎙️ Entrada de audio:", font=("Consolas", 9),
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
            self.root, height=4, font=("Consolas", 10),
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
        tk.Label(self.root, text="💡 RESPUESTA SUGERIDA",
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
            self.root, height=6, font=("Consolas", 8),
            bg=BG3, fg=GRIS, relief="flat", padx=8, pady=4,
            wrap="word", state="disabled"
        )
        self.txt_log.pack(fill="x", padx=12, pady=(0,10))
        
        # Copiar respuesta con doble clic
        self.txt_respuesta.bind("<Double-Button-1>", self.copiar_respuesta)
        self.root.bind("<Escape>", lambda e: self.root.iconify())
        
        # Tip
        tk.Label(self.root, text="💡 Doble clic en respuesta para copiar | ESC para minimizar",
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
            
            # Seleccionar VB-Cable automáticamente si existe
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
                    self._set_texto(self.txt_respuesta, "⏳ Buscando respuesta...", "#64748b")
                
                elif tipo == "respuesta":
                    respuesta, proveedor, pregunta, elapsed = datos
                    self._set_texto(self.txt_respuesta, respuesta, "#00ff88")
                    self.lbl_proveedor.config(text=f"{proveedor} | {elapsed:.1f}s")
                    self._log(f"Respuesta via {proveedor} en {elapsed:.1f}s")
                
                elif tipo == "fuentes":
                    fuentes = datos
                    self.lbl_fuentes.config(text=f"Fuentes: {' | '.join(fuentes)}")
                
                elif tipo == "nivel_audio":
                    self.actualizar_nivel_audio(datos)
        
        except queue.Empty:
            pass
        except Exception as e:
            pass
        
        # Re-programar para el siguiente tick
        self.root.after(80, self.procesar_cola)
    
    def ejecutar(self):
        self.root.after(100, self.procesar_cola)
        self.root.mainloop()


# ════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  ⚡ RAG ASISTENTE - Iniciando...")
    print("="*60)
    print()
    
    app = InterfazFlotante()
    app.ejecutar()
