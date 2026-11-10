# !pip install sentence-transformers chromadb

import data_treatment
import chromadb
from chromadb.utils import embedding_functions

def db_init() -> chromadb.Collection:
    print("[DBINIT:LOG] Inicializado ChromaDB.")
    
    # BERTimbau como modelo de embeddings.
    # Sentence transformer aplica pooling à saída do modelo para obtenção do vetor de frase.
    model_name = "neuralmind/bert-base-portuguese-cased"
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    
    chroma_client = chromadb.PersistentClient(path = "../chroma_db_reviews")

    collection_name = "reviews_bertimbau"
            
    try:
        chroma_client.delete_collection(name = collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = chroma_client.create_collection(name = collection_name, embedding_function = embedding_function)
    
    print("[DBINIT:LOG] ChromaDB inicializado com sucesso.")
    
    return collection

collection = db_init()
try:
    df_reviews = data_treatment.get_treated_data()

    # IDs únicos em formato string para o ChromaDB.
    documents = df_reviews['rag_document'].tolist()
    ids = [f"review_{i}" for i in range(len(df_reviews))]

    metadados = []
    for _, row in df_reviews.iterrows():
        metadados.append({
            "review_score": int(row['review_score']),
            "order_id": str(row['order_id']),
            "purchase_date": str(row['order_purchase_timestamp'])
        })
    
    # Inserção no banco em batches para maior eficiência,
    # e para que seja possível fornecer updates do progresso.
    print(f"[IDX:LOG] Iniciando indexação em batches.")
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        
        collection.add(ids=ids[i:end_idx], metadatas=metadados[i:end_idx], documents=documents[i:end_idx])
        print(f"[IDX:LOG] Batch {i} até {end_idx} indexado.")
    print("[IDX:LOG] Indexação concluída com sucesso!")
except Exception as e:
    print(f"[IDX:ERR] Falha na indexação dos dados: {str(e)}")