@echo off
echo Indexando PDFs...
call venv\Scripts\activate.bat
python indexar_pdfs.py
pause
