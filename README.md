# ⚡ RAG ASISTENTE - Guía Completa

Asistente en tiempo real que escucha tus reuniones de Teams y responde
preguntas usando tus documentos PDF como base de conocimiento.

---

## 🏗️ Arquitectura

```
Teams (audio) → VB-Cable → faster-whisper (local) → ChromaDB RAG (local)
                                                            ↓
                                          Groq API → OpenRouter → Gemini
                                                            ↓
                                              Ventana flotante en pantalla
```

---

## 📋 Instalación paso a paso

### Paso 1: VB-Audio Virtual Cable
1. Descarga desde: https://vb-audio.com/Cable/
2. Instala y reinicia Windows
3. En Teams → Configuración → Dispositivos → Altavoz: selecciona "CABLE Input"
4. Pon auriculares para escuchar Teams desde "CABLE Output"

### Paso 2: API Keys (todas gratuitas)

| Servicio | URL | Velocidad | Plan gratis |
|----------|-----|-----------|-------------|
| **Groq** | https://console.groq.com | ~500ms ⚡ | Sí |
| **OpenRouter** | https://openrouter.ai | ~2s | Sí (modelos free) |
| **Gemini** | https://aistudio.google.com | ~2s | Sí, muy generoso |

Copia las keys en el archivo `.env`

### Paso 3: Instalar dependencias
```
Doble clic en: 1_INSTALAR.bat
```
**Nota:** Se creará un entorno virtual aislado (`venv/`) para las dependencias.

### Paso 4: Indexar tus PDFs
1. Copia tus PDFs a la carpeta `pdfs\`
2. Doble clic en: `2_INDEXAR_PDFS.bat`
3. Espera que termine (primera vez descarga el modelo ~400MB)

### Paso 5: Iniciar asistente
```
Doble clic en: 3_INICIAR_ASISTENTE.bat
```

---

## 🎯 Uso durante la reunión

1. Abre Teams y únete a la reunión
2. Inicia el asistente (`3_INICIAR_ASISTENTE.bat`)
3. Selecciona "CABLE Output" como dispositivo de entrada
4. Presiona **▶ INICIAR**
5. La ventana flotante aparece sobre Teams
6. Cuando el tutor haga una pregunta → la respuesta aparece en ~2-3 segundos
7. **Doble clic** en la respuesta para copiarla al portapapeles

---

## ⚙️ Configuración avanzada (.env)

```env
# Modelo Whisper: tiny (más rápido) | base (recomendado) | small (más preciso)
WHISPER_MODEL=base

# Fragmentos RAG a recuperar (más = mejor contexto, más lento)
RAG_TOP_K=4

# Tamaño de fragmentos de texto
CHUNK_SIZE=800
```

---

## 🔧 Solución de problemas

**No detecta audio:**
- Verifica que VB-Cable esté instalado
- En Teams, configura el altavoz como "CABLE Input (VB-Audio)"
- Selecciona "CABLE Output" en el combo del asistente

**Respuestas lentas:**
- Cambia `WHISPER_MODEL=tiny` en .env para transcripción más rápida
- Groq debería responder en <1 segundo

**No encuentra información:**
- Re-ejecuta `2_INDEXAR_PDFS.bat` después de agregar PDFs
- Aumenta `RAG_TOP_K=6` en .env

**Error de API key:**
- Verifica que las keys estén correctamente copiadas en `.env`
- Sin espacios antes/después de la key

---

## 📁 Estructura del proyecto

```
rag-asistente/
├── venv/               ← Entorno virtual (auto-generado)
├── pdfs/               ← Coloca tus PDFs aquí
├── db/                 ← Base de datos vectorial (auto-generada)
├── logs/               ← Logs del sistema
├── .env                ← Configuración y API keys
├── indexar_pdfs.py     ← Script de indexación
├── asistente.py        ← Aplicación principal
├── requirements.txt    ← Dependencias Python
├── 1_INSTALAR.bat      ← Instalador automático
├── 2_INDEXAR_PDFS.bat  ← Indexar documentos
├── 3_INICIAR_ASISTENTE.bat ← Lanzar asistente
└── TEST_APIS.bat       ← Probar APIs configuradas
```
