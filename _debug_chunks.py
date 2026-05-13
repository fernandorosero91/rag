"""Debug: buscar chunks que contienen 'Entorno Saludable'"""
import chromadb
from sentence_transformers import SentenceTransformer

modelo = SentenceTransformer("BAAI/bge-m3")
cliente = chromadb.PersistentClient(path="./db")
coleccion = cliente.get_collection("documentos_rag")

# Buscar por texto directo
results = coleccion.get(include=["documents", "metadatas"])

print(f"Total chunks: {len(results['documents'])}")
print("\n--- Chunks que contienen 'Entorno Saludable' ---")
for i, doc in enumerate(results["documents"]):
    if "entorno saludable" in doc.lower() or "sg-sst" in doc.lower():
        print(f"\nChunk {i}:")
        print(doc[:500])
        print("---")

print("\n--- Chunks que contienen 'Transición Justa' ---")
for i, doc in enumerate(results["documents"]):
    if "transición justa" in doc.lower():
        print(f"\nChunk {i}:")
        print(doc[:500])
        print("---")
