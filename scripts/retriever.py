# !pip install google-genai

import os
import time
import json
import chromadb
from google import genai
from sentence_transformers import CrossEncoder

with open("../gemini_api_key.txt", 'r') as file:
    os.environ["GEMINI_API_KEY"] = file.read()

rerank_model = CrossEncoder('BAAI/bge-reranker-base')

def get_query_intent(user_query):

    client = genai.Client()
    
    prompt_parser = (
        "Você é um classificador de intenções para um sistema de e-commerce.\n"
        "Analise a pergunta do usuário e determine se há alguma restrição explícita ou implícita de nota de avaliação.\n"
        "As notas variam de 1 a 5.\n\n"
        f"Pergunta: '{user_query}'\n\n"
        "Retorne APENAS um objeto JSON válido (sem markdown, sem crases) com o seguinte formato:\n"
        "{\n"
        '  "filtra_nota": true ou false,\n'
        '  "min_score": 1 a 5 (ou null se não houver restrição),\n'
        '  "max_score": 1 a 5 (ou null se não houver restrição)\n'
        "}\n"
        "Exemplos:\n"
        "- 'principais problemas nas notas baixas' -> {\"filtra_nota\": true, \"min_score\": 1, \"max_score\": 2}\n"
        "- 'o que os clientes elogiam' -> {\"filtra_nota\": true, \"min_score\": 4, \"max_score\": 5}\n"
        "- 'como foi a entrega' -> {\"filtra_nota\": false, \"min_score\": null, \"max_score\": null}"
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt_parser
        )
        
        # Limpeza do JSON retornado, para consistência na formatação.
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data_filter = json.loads(clean_text)
        
        # Converte para o formato de operador do ChromaDB.
        if data_filter.get("filtra_nota"):
            min_s = data_filter.get("min_score")
            max_s = data_filter.get("max_score")
            
            if min_s and max_s:
                return {"$and": [{"review_score": {"$gte": min_s}}, {"review_score": {"$lte": max_s}}]}
            elif min_s:
                return {"review_score": {"$gte": min_s}}
            elif max_s:
                return {"review_score": {"$lte": max_s}}
                
    except Exception as e:
        print(f"Erro ao extrair intenção via LLM: {e}. Seguindo sem filtro rígido.")
    
    # Caso não seja identificada restrição, ou ocorra falha na request ao LLM.
    return None

def retrieve_evidences(collection, user_query, n_starting_candidates=15, n_final=5):
    
    dynamic_filter = get_query_intent(user_query)
    print(f"Filtro de metadados aplicado automaticamente: {dynamic_filter}")
    
    results = collection.query(query_texts=[user_query], n_results=n_starting_candidates, where=dynamic_filter)
    
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

def get_llm_response(complete_prompt):
    client = genai.Client()
        
    attempt = 0
    max_retries = 3
    sleep_time = 5
    
    while attempt < max_retries:
        try:
            response = client.models.generate_content(model='gemini-3.6-flash',contents=complete_prompt)
            return response.text
            
        except Exception as e:
            attempt += 1
            print(f"Aviso: Modelo ocupado ou limite atingido (Tentativa {attempt}/{max_retries}). Erro: {e}")
            
            if attempt >= max_retries:
                raise Exception("Número máximo de tentativas atingido. O serviço permaneceu indisponível devido à alta carga.")
            
            print(f"Aguardando {sleep_time} segundos antes de tentar novamente...")
            time.sleep(sleep_time)
            sleep_time *= 2

def get_query_reply(user_query, n_results=5):
    chroma_client = chromadb.PersistentClient(path="../chroma_db_reviews")
    collection = chroma_client.get_collection(name="reviews_bertimbau")
    
    context_text, audit_evidences = retrieve_evidences(collection, user_query, n_starting_candidates=n_results, n_final=15)
    
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
    
    try:
        return get_llm_response(complete_prompt), audit_evidences
    except Exception as e:
        print(f"Não foi possível obter uma resposta para a pergunta feita. Erro: {e}")
    

pergunta = "O que os clientes relatam sobre problemas relacionados à entrega?"

resposta, fontes = get_query_reply(pergunta, n_results=50)

print("=== RESPOSTA DO RAG ===")
print(resposta)
print("\n=== EVIDÊNCIAS UTILIZADAS ===")
for fonte in fontes:
    print(f"- Evidência {fonte['evidencia_id']}: Pedido {fonte['order_id']} | Nota: {fonte['review_score']} estrelas | Score: {fonte['rerank_score']}")