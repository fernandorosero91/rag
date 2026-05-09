@echo off
echo ============================================================
echo   RAG ASISTENTE - INSTALADOR AUTOMATICO
echo ============================================================
echo.

:: Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado.
    echo Descarga Python 3.10+ desde https://python.org
    pause
    exit /b 1
)

echo [1/5] Python encontrado OK
python --version

:: Crear entorno virtual si no existe
echo.
echo [2/5] Creando entorno virtual...
if not exist "venv\" (
    python -m venv venv
    echo Entorno virtual creado
) else (
    echo Entorno virtual ya existe
)

:: Activar entorno virtual
echo.
echo [3/5] Activando entorno virtual...
call venv\Scripts\activate.bat

:: Actualizar pip
echo.
echo [4/5] Actualizando pip...
python -m pip install --upgrade pip --quiet

:: Instalar dependencias
echo.
echo [5/5] Instalando dependencias (puede tardar 3-5 minutos)...
echo       Se descargara faster-whisper, sentence-transformers, etc.
echo.
pip install faster-whisper chromadb sentence-transformers pymupdf sounddevice numpy requests python-dotenv

if errorlevel 1 (
    echo.
    echo [ERROR] Fallo la instalacion de dependencias.
    pause
    exit /b 1
)

echo.
echo Verificando instalacion...
python -c "import faster_whisper; import chromadb; import sentence_transformers; import fitz; import sounddevice; print('Todas las dependencias OK')"

if errorlevel 1 (
    echo [WARNING] Algunas dependencias pueden tener problemas.
) else (
    echo.
    echo ============================================================
    echo   INSTALACION COMPLETADA EXITOSAMENTE
    echo ============================================================
    echo.
    echo PROXIMOS PASOS:
    echo.
    echo 1. Edita el archivo .env con tus API keys
    echo    - GROQ:        https://console.groq.com (gratis)
    echo    - OPENROUTER:  https://openrouter.ai (gratis)
    echo    - GEMINI:      https://aistudio.google.com (gratis)
    echo.
    echo 2. Copia tus PDFs a la carpeta 'pdfs\'
    echo.
    echo 3. Ejecuta:  python indexar_pdfs.py
    echo.
    echo 4. Ejecuta:  python asistente.py
    echo.
)

pause
