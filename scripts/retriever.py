# !pip install google-genai

import os
import time
import chromadb
from google import genai
from sentence_transformers import CrossEncoder

with open("../gemini_api_key.txt", 'r') as file:
    os.environ["GEMINI_API_KEY"] = file.read()

rerank_model = CrossEncoder('BAAI/bge-reranker-base')

def retrieve_evidences(collection, user_query, n_starting_candidates=15, n_final=5):
    
    results = collection.query(query_texts=[user_query], n_results=n_starting_candidates)
    
    documents = results['documents'][0]
    metadata = results['metadatas'][0]
    
    if not documents:
        return "", []
        
    pairs = [[user_query, doc] for doc in documents]
    
    # Reclassificação da relevância dos documentos obtidos conforme modelo de reranking,
    # mantendo apenas o top N final.
    scores = rerank_model.predict(pairs)
    combined_data = list(zip(documents, metadata, scores))
    combined_data.sort(key=lambda x: x[2], reverse=True) 
    top_results = combined_data[:n_final]
    
    context_text = ""
    audit_evidences = []
    
    for i, (doc, meta, score) in enumerate(top_results):
        
        context_text += f"--- Evidência [{i+1}] ---\n{doc}\n\n"
        audit_evidences.append({
            "evidencia_id": i + 1,
            "order_id": meta.get('order_id'),
            "review_score": meta.get('review_score'),
            "purchase_date": meta.get('purchase_date'),
            "rerank_score": float(score)
        })
    
    if not audit_evidences:
        return "", []
        
    return context_text, audit_evidences

def get_query_reply(user_query, n_results=5):
    chroma_client = chromadb.PersistentClient(path="../chroma_db_reviews")
    collection = chroma_client.get_collection(name="reviews_bertimbau")
    
    context_text, audit_evidences = retrieve_evidences(collection, user_query, n_starting_candidates=20, n_final=10)
    
    if not context_text:
        return "Não foram encontradas avaliações relevantes para responder à sua pergunta na base de conhecimento.", []
        
    system_prompt = (
        "Você é um assistente sênior de Inteligência de Negócios (Voice of Customer Intelligence) "
        "especializado em analisar avaliações de e-commerce.\n"
        "Sua missão é responder à pergunta do gestor baseando-se **estritamente e exclusivamente** "
        "nas avaliações de clientes fornecidas no contexto abaixo.\n\n"
        "Diretrizes obrigatórias:\n"
        "1. Não invente fatos, opiniões ou dados que não estejam amparados pelas evidências.\n"
        "2. Sempre cite as evidências correspondentes (ex: citando a Evidência [X], a nota ou o período) para sustentar seus argumentos.\n"
        "3. Caso as evidências recuperadas sejam insuficientes, inconclusivas ou a pergunta esteja totalmente fora do escopo "
        "da base de conhecimento, responda obrigatoriamente: "
        "'Não possuo informações suficientes nas avaliações de clientes para responder a esta pergunta com segurança.'"
    )
    
    complete_prompt = (
        f"{system_prompt}\n\n"
        f"Contexto (Avaliações dos Clientes):\n{context_text}\n"
        f"Pergunta do Gestor: {user_query}"
    )
    
    client = genai.Client()
    
    attempt = 0
    max_retries = 3
    sleep_time = 5
    
    while attempt < max_retries:
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=complete_prompt,
            )
            return response.text, audit_evidences
            
        except Exception as e:
            attempt += 1
            print(f"Aviso: Modelo ocupado ou limite atingido (Tentativa {attempt}/{max_retries}). Erro: {e}")
            
            if attempt >= max_retries:
                raise Exception("Número máximo de tentativas atingido. O serviço permaneceu indisponível devido à alta carga.")
            
            print(f"Aguardando {sleep_time} segundos antes de tentar novamente...")
            time.sleep(sleep_time)
            sleep_time *= 2


pergunta = "Quais são os principais problemas relatados pelos clientes relacionados ao atraso na entrega?"

resposta, fontes = get_query_reply(pergunta, n_results=20)

print("=== RESPOSTA DO RAG ===")
print(resposta)
print("\n=== EVIDÊNCIAS UTILIZADAS ===")
for fonte in fontes:
    print(f"- Evidência {fonte['evidencia_id']}: Pedido {fonte['order_id']} | Nota: {fonte['review_score']} estrelas | Score: {fonte['rerank_score']}")