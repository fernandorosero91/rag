"""
================================================================
INDEXADOR DE PDFs - RAG ASISTENTE v2.0
================================================================
Mejoras:
  - Chunking inteligente por oraciones (no corta a mitad de frase)
  - Modelo de embeddings BGE-M3 (1024-dim, superior en español)
  - Índice BM25 para búsqueda híbrida (keyword + semántica)
  - Metadata limpia (página como dato, no como texto)

Uso: python indexar_pdfs.py
================================================================
"""

import os
import sys
import re
import json
import time
import math
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# ── Dependencias ──────────────────────────────────────────────
try:
    import fitz  # pymupdf
    import chromadb
    from sentence_transformers import SentenceTransformer
    print("✅ Dependencias OK")
except ImportError as e:
    print(f"❌ Falta instalar: {e}")
    print("\nEjecuta: pip install pymupdf chromadb sentence-transformers")
    sys.exit(1)

# ── Configuración ─────────────────────────────────────────────
PDF_FOLDER    = os.getenv("PDF_FOLDER", "./pdfs")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))      # Reducido para mayor precisión
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))    # Overlap por oraciones
DB_PATH       = "./db"
BM25_PATH     = "./db/bm25_index.json"

# Modelo de embeddings — BGE-M3: 1024-dim, excelente en español
EMBED_MODEL = "BAAI/bge-m3"


# ════════════════════════════════════════════════════════════
#  CHUNKING INTELIGENTE POR ORACIONES
# ════════════════════════════════════════════════════════════
def dividir_en_oraciones(texto: str) -> list[str]:
    """Divide texto en oraciones respetando abreviaturas comunes."""
    # Patrón: punto/signo seguido de espacio y mayúscula o fin de línea
    oraciones = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])', texto)
    # También dividir por saltos de línea dobles (párrafos)
    resultado = []
    for oracion in oraciones:
        partes = oracion.split('\n\n')
        for parte in partes:
            limpia = parte.strip()
            if limpia:
                resultado.append(limpia)
    return resultado


def chunking_por_oraciones(texto: str, max_chars: int = 512, overlap_oraciones: int = 1) -> list[str]:
    """
    Divide texto en chunks respetando límites de oraciones.
    Nunca corta a mitad de frase.
    """
    oraciones = dividir_en_oraciones(texto)
    
    if not oraciones:
        return []
    
    chunks = []
    i = 0
    
    while i < len(oraciones):
        chunk_actual = ""
        oraciones_en_chunk = 0
        inicio_chunk = i
        
        while i < len(oraciones):
            candidato = chunk_actual + (" " if chunk_actual else "") + oraciones[i]
            
            if len(candidato) > max_chars and chunk_actual:
                # El chunk ya está lleno, no agregar más
                break
            
            chunk_actual = candidato
            oraciones_en_chunk += 1
            i += 1
            
            # Si una sola oración excede max_chars, la incluimos completa
            if len(chunk_actual) > max_chars:
                break
        
        if chunk_actual.strip():
            chunks.append(chunk_actual.strip())
        
        # Overlap: retroceder N oraciones
        if overlap_oraciones > 0 and i < len(oraciones):
            i = max(inicio_chunk + 1, i - overlap_oraciones)
    
    return chunks


# ════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE TEXTO
# ════════════════════════════════════════════════════════════
def extraer_texto_pdf(pdf_path: str) -> list[dict]:
    """Extrae texto de un PDF con metadata de página limpia."""
    doc = fitz.open(pdf_path)
    paginas = []
    
    for num_pagina, pagina in enumerate(doc, 1):
        texto = pagina.get_text("text")
        if texto.strip():
            paginas.append({
                "texto": texto.strip(),
                "pagina": num_pagina
            })
    
    doc.close()
    
    if not paginas:
        return []
    
    # Unir todo el texto para chunking coherente
    texto_completo = "\n\n".join(p["texto"] for p in paginas)
    
    # Crear mapa de posición → página
    posicion_pagina = []
    pos = 0
    for p in paginas:
        largo = len(p["texto"]) + 2  # +2 por \n\n
        posicion_pagina.append((pos, pos + largo, p["pagina"]))
        pos += largo
    
    # Chunking inteligente
    chunks = chunking_por_oraciones(texto_completo, max_chars=CHUNK_SIZE, overlap_oraciones=1)
    
    # Asignar página a cada chunk
    fragmentos = []
    pos_actual = 0
    
    for idx, chunk in enumerate(chunks):
        # Encontrar en qué posición del texto completo está este chunk
        chunk_pos = texto_completo.find(chunk[:50], max(0, pos_actual - 100))
        if chunk_pos == -1:
            chunk_pos = pos_actual
        
        # Determinar página
        pagina_chunk = 1
        for inicio, fin, pag in posicion_pagina:
            if inicio <= chunk_pos < fin:
                pagina_chunk = pag
                break
        
        pos_actual = chunk_pos + len(chunk)
        
        fragmentos.append({
            "texto": chunk,
            "fuente": Path(pdf_path).name,
            "pagina": pagina_chunk,
            "indice": idx
        })
    
    return fragmentos


# ════════════════════════════════════════════════════════════
#  ÍNDICE BM25 (búsqueda por keywords)
# ════════════════════════════════════════════════════════════
def tokenizar(texto: str) -> list[str]:
    """Tokenización simple para BM25."""
    texto = texto.lower()
    texto = re.sub(r'[^\w\sáéíóúñü]', ' ', texto)
    tokens = texto.split()
    # Filtrar stopwords muy comunes
    stopwords = {'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'del', 'un', 'una',
                 'que', 'es', 'se', 'por', 'con', 'para', 'al', 'lo', 'como', 'su',
                 'más', 'o', 'este', 'ya', 'entre', 'cuando', 'muy', 'sin', 'sobre',
                 'ser', 'también', 'me', 'hasta', 'hay', 'donde', 'le', 'todo', 'nos'}
    return [t for t in tokens if t not in stopwords and len(t) > 2]


def construir_indice_bm25(fragmentos: list[dict]) -> dict:
    """Construye un índice BM25 serializable."""
    N = len(fragmentos)
    
    # Tokenizar todos los documentos
    docs_tokens = []
    for f in fragmentos:
        tokens = tokenizar(f["texto"])
        docs_tokens.append(tokens)
    
    # Calcular IDF
    df = Counter()  # document frequency
    for tokens in docs_tokens:
        for token in set(tokens):
            df[token] += 1
    
    # Largo promedio de documentos
    avgdl = sum(len(t) for t in docs_tokens) / max(N, 1)
    
    return {
        "N": N,
        "avgdl": avgdl,
        "df": dict(df),
        "docs_tokens": docs_tokens,
        "doc_lengths": [len(t) for t in docs_tokens]
    }


# ════════════════════════════════════════════════════════════
#  INDEXACIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════
def indexar_todos_los_pdfs():
    """Indexa todos los PDFs con embeddings BGE-M3 + índice BM25."""
    
    print("\n" + "="*60)
    print("  📚 INDEXADOR DE PDFs - RAG ASISTENTE v2.0")
    print("="*60)
    print(f"  Modelo: {EMBED_MODEL}")
    print(f"  Chunk size: {CHUNK_SIZE} chars (por oraciones)")
    print(f"  Búsqueda: Híbrida (vectorial + BM25)")
    
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
    print(f"\n🧠 Cargando modelo de embeddings BGE-M3...")
    print(f"   (Primera vez descarga ~2.2 GB, luego es instantáneo)")
    
    t0 = time.time()
    modelo = SentenceTransformer(EMBED_MODEL)
    print(f"   ✅ Modelo cargado en {time.time()-t0:.1f}s")
    print(f"   Dimensiones: {modelo.get_sentence_embedding_dimension()}")
    
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
    todos_fragmentos = []
    
    for pdf_path in pdfs:
        print(f"\n📖 Procesando: {pdf_path.name}")
        
        t0 = time.time()
        fragmentos = extraer_texto_pdf(str(pdf_path))
        print(f"   • {len(fragmentos)} fragmentos extraídos (chunking por oraciones)")
        
        if not fragmentos:
            print(f"   ⚠️  No se pudo extraer texto (¿PDF escaneado?)")
            continue
        
        # Generar embeddings en lotes
        textos   = [f["texto"] for f in fragmentos]
        ids      = [f"{pdf_path.stem}_{f['indice']}" for f in fragmentos]
        metadatas = [{"fuente": f["fuente"], "pagina": f["pagina"], "indice": f["indice"]} for f in fragmentos]
        
        print(f"   • Generando embeddings BGE-M3...", end="", flush=True)
        embeddings = modelo.encode(textos, show_progress_bar=False, normalize_embeddings=True).tolist()
        print(f" ✅")
        
        # Guardar en ChromaDB
        # ChromaDB tiene límite de batch, dividir si es necesario
        batch_size = 500
        for b_start in range(0, len(ids), batch_size):
            b_end = min(b_start + batch_size, len(ids))
            coleccion.add(
                ids=ids[b_start:b_end],
                embeddings=embeddings[b_start:b_end],
                documents=textos[b_start:b_end],
                metadatas=metadatas[b_start:b_end]
            )
        
        todos_fragmentos.extend(fragmentos)
        print(f"   • Indexado en {time.time()-t0:.1f}s")
    
    # Construir índice BM25
    print(f"\n🔤 Construyendo índice BM25 (búsqueda por keywords)...")
    indice_bm25 = construir_indice_bm25(todos_fragmentos)
    
    # Guardar BM25 + textos para búsqueda híbrida
    bm25_data = {
        "indice": indice_bm25,
        "textos": [f["texto"] for f in todos_fragmentos],
        "metadatas": [{"fuente": f["fuente"], "pagina": f["pagina"]} for f in todos_fragmentos]
    }
    
    os.makedirs(os.path.dirname(BM25_PATH), exist_ok=True)
    with open(BM25_PATH, "w", encoding="utf-8") as f:
        json.dump(bm25_data, f, ensure_ascii=False)
    
    print(f"   ✅ Índice BM25 guardado ({len(todos_fragmentos)} documentos)")
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"  ✅ INDEXACIÓN COMPLETADA (v2.0)")
    print(f"{'='*60}")
    print(f"  📊 PDFs procesados:    {len(pdfs)}")
    print(f"  🧩 Total fragmentos:   {len(todos_fragmentos)}")
    print(f"  🧠 Embeddings:         BGE-M3 (1024-dim)")
    print(f"  🔤 BM25:               {len(indice_bm25['df'])} términos únicos")
    print(f"  💾 Base de datos en:   {DB_PATH}/")
    print(f"\n  ✅ Ya puedes ejecutar: python asistente.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    indexar_todos_los_pdfs()
