"""
================================================================
INDEXADOR DE PDFs - RAG ASISTENTE
================================================================
Ejecuta este script UNA VEZ para indexar tus PDFs.
Luego cada vez que agregues PDFs nuevos, ejecútalo de nuevo.

Uso: python indexar_pdfs.py
================================================================
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Dependencias ──────────────────────────────────────────────
try:
    import fitz  # pymupdf
    import chromadb
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer
    print("✅ Dependencias OK")
except ImportError as e:
    print(f"❌ Falta instalar: {e}")
    print("\nEjecuta: pip install pymupdf chromadb sentence-transformers")
    sys.exit(1)

# ── Configuración ─────────────────────────────────────────────
PDF_FOLDER   = os.getenv("PDF_FOLDER", "./pdfs")
CHUNK_SIZE   = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP= int(os.getenv("CHUNK_OVERLAP", 100))
DB_PATH      = "./db"

# ── Modelo de embeddings (local, liviano, bueno en español) ───
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def extraer_texto_pdf(pdf_path: str) -> list[dict]:
    """Extrae texto de un PDF y lo divide en fragmentos."""
    doc = fitz.open(pdf_path)
    texto_completo = ""
    
    for num_pagina, pagina in enumerate(doc, 1):
        texto = pagina.get_text("text")
        if texto.strip():
            texto_completo += f"\n[Página {num_pagina}]\n{texto}"
    
    doc.close()
    
    # Dividir en fragmentos con overlap
    fragmentos = []
    inicio = 0
    indice = 0
    
    while inicio < len(texto_completo):
        fin = inicio + CHUNK_SIZE
        fragmento = texto_completo[inicio:fin]
        
        if fragmento.strip():
            fragmentos.append({
                "texto": fragmento.strip(),
                "fuente": Path(pdf_path).name,
                "indice": indice
            })
            indice += 1
        
        inicio = fin - CHUNK_OVERLAP
    
    return fragmentos


def indexar_todos_los_pdfs():
    """Indexa todos los PDFs en la carpeta configurada."""
    
    print("\n" + "="*60)
    print("  📚 INDEXADOR DE PDFs - RAG ASISTENTE")
    print("="*60)
    
    # Verificar carpeta de PDFs
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"\n⚠️  Carpeta '{PDF_FOLDER}' creada.")
        print(f"   Coloca tus PDFs ahí y ejecuta este script de nuevo.")
        return
    
    pdfs = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    if not pdfs:
        print(f"\n⚠️  No se encontraron PDFs en '{PDF_FOLDER}'")
        print(f"   Coloca tus PDFs ahí y ejecuta este script de nuevo.")
        return
    
    print(f"\n📄 PDFs encontrados: {len(pdfs)}")
    for pdf in pdfs:
        size_mb = pdf.stat().st_size / (1024*1024)
        print(f"   • {pdf.name} ({size_mb:.1f} MB)")
    
    # Cargar modelo de embeddings
    print(f"\n🧠 Cargando modelo de embeddings...")
    print(f"   (Primera vez puede tardar ~2 minutos en descargar)")
    
    t0 = time.time()
    modelo = SentenceTransformer(EMBED_MODEL)
    print(f"   ✅ Modelo cargado en {time.time()-t0:.1f}s")
    
    # Inicializar ChromaDB
    print(f"\n🗄️  Inicializando base de datos vectorial...")
    cliente = chromadb.PersistentClient(path=DB_PATH)
    
    # Eliminar colección anterior si existe (re-indexar)
    try:
        cliente.delete_collection("documentos_rag")
        print("   ♻️  Base de datos anterior eliminada (re-indexando)")
    except:
        pass
    
    coleccion = cliente.create_collection(
        name="documentos_rag",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Procesar cada PDF
    total_fragmentos = 0
    
    for pdf_path in pdfs:
        print(f"\n📖 Procesando: {pdf_path.name}")
        
        t0 = time.time()
        fragmentos = extraer_texto_pdf(str(pdf_path))
        print(f"   • {len(fragmentos)} fragmentos extraídos")
        
        if not fragmentos:
            print(f"   ⚠️  No se pudo extraer texto (¿PDF escaneado?)")
            continue
        
        # Generar embeddings en lotes
        textos   = [f["texto"] for f in fragmentos]
        fuentes  = [f["fuente"] for f in fragmentos]
        indices  = [f["indice"] for f in fragmentos]
        ids      = [f"{pdf_path.stem}_{i}" for i in indices]
        
        print(f"   • Generando embeddings...", end="", flush=True)
        embeddings = modelo.encode(textos, show_progress_bar=False).tolist()
        print(f" ✅")
        
        # Guardar en ChromaDB
        coleccion.add(
            ids=ids,
            embeddings=embeddings,
            documents=textos,
            metadatas=[{"fuente": f, "indice": i} for f, i in zip(fuentes, indices)]
        )
        
        total_fragmentos += len(fragmentos)
        print(f"   • Indexado en {time.time()-t0:.1f}s")
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"  ✅ INDEXACIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"  📊 PDFs procesados:   {len(pdfs)}")
    print(f"  🧩 Total fragmentos:  {total_fragmentos}")
    print(f"  💾 Base de datos en:  {DB_PATH}/")
    print(f"\n  ✅ Ya puedes ejecutar: python asistente.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    indexar_todos_los_pdfs()
