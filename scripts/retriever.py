# !pip install google-genai

import os
import warnings
import logging

##### Configurações adicionais de logging e váriaveis de ambiente para deixar o terminal mais limpo quando executando.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)
#####

import time
import json
import chromadb
from google import genai
from sentence_transformers import CrossEncoder

rerank_model = CrossEncoder('BAAI/bge-reranker-base')

filter_dict = {
    "filtra_nota": {
        "coluna": "review_score",
        "min": "min_score",
        "max": "max_score"
        } ,
    "filtra_valor": {
        "coluna": "payment_value",
        "min": "min_value",
        "max": "max_value"
    },
    "filtra_estado": {
        "coluna": "customer_state",
        "valor": "estado"
    }
}

def get_query_intent(user_query):

    client = genai.Client()
    
    prompt_parser = (
        "Você é um classificador de intenções para um sistema de e-commerce.\n"
        "Analise a pergunta do usuário e determine se há alguma restrição explícita ou implícita de nota de avaliação, valor da compra ou de estado/região do comprador..\n"
        "As notas variam de 1 a 5. Para valores, considere:\n"
        "- Compras baratas / baixo valor: até 50.\n"
        "- Compras caras / alto valor: acima de 500.\n"
        "- Se houver um valor específico (ex: 'acima de 150 reais'), extraia o número correspondente.\n\n"
        "Para estados, extraia a sigla de duas letras em maiúsculo (ex: SP, RJ, MG), e retorne uma lista contendo a sigla correspondente. Se citar o nome completo, converta para a sigla correspondente.\n\n"
        "Para perguntas que envolvam regiões do Brasil, retorne uma lista com as siglas de duas letras em maiúsculo que compõem a região correspondente.\n\n"
        f"Pergunta: '{user_query}'\n\n"
        "Retorne APENAS um objeto JSON válido (sem markdown, sem crases) com o seguinte formato:\n"
        "{\n"
        '  "filtra_nota": true ou false,\n'
        '  "min_score": 1 a 5 (ou null se não houver restrição),\n'
        '  "max_score": 1 a 5 (ou null se não houver restrição),\n'
        '  "filtra_valor": true ou false,\n'
        '  "min_value": maior que zero e menor que max_value (ou null se não houver restrição),\n'
        '  "max_value": maior que min_value (ou null se não houver restrição)\n'
        '  "filtra_estado": true ou false,\n'
        '  "estado": ["SIGLA_DO_ESTADO"] (ou null)\n'
        "}\n"
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',
            contents=prompt_parser
        )
        
        # Limpeza do JSON retornado, para consistência na formatação.
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data_filter = json.loads(clean_text)
        
        identified_filters = []
        
        for key, value in filter_dict.items():
            if data_filter.get(key):
                data_column = value['coluna']
                if data_column == 'customer_state':
                    identified_filters.append({data_column: {"$in": data_filter.get(value['valor'])}})
                else:
                    
                    min_s = data_filter.get(value['min'])
                    max_s = data_filter.get(value['max'])
                    
                    if min_s and max_s:
                        identified_filters.append({"$and": [{data_column: {"$gte": min_s}}, {data_column: {"$lte": max_s}}]})
                    elif min_s:
                        identified_filters.append({data_column: {"$gte": min_s}})
                    elif max_s:
                        identified_filters.append({data_column: {"$lte": max_s}})
        
        if len(identified_filters) > 1:
            dynamic_filter = {"$and": identified_filters}
        elif len(identified_filters) == 1:
            dynamic_filter = identified_filters[0]
        else:
            dynamic_filter = None
            
    except Exception as e:
        print(f"Erro ao extrair intenção via LLM: {e}. Seguindo sem filtro rígido.")
    
    return dynamic_filter

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
            "payment_value": meta.get('payment_value'),
            "customer_state": meta.get('customer_state'),
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
            response = client.models.generate_content(model='gemini-3.5-flash',contents=complete_prompt)
            return response.text
            
        except Exception as e:
            attempt += 1
            print(f"Aviso: Modelo ocupado ou limite atingido (Tentativa {attempt}/{max_retries}). Erro: {e}")
            
            if attempt >= max_retries:
                raise Exception("Número máximo de tentativas atingido. O serviço permaneceu indisponível devido à alta carga.")
            
            print(f"Aguardando {sleep_time} segundos antes de tentar novamente...")
            time.sleep(sleep_time)
            sleep_time *= 2

def get_query_reply(user_query, n_results):
    
    if not os.environ.get("GEMINI_API_KEY"):
        try:
            with open("../gemini_api_key.txt", 'r') as file:
                os.environ["GEMINI_API_KEY"] = file.read().strip()
        except FileNotFoundError:
            print(f"Chave para API Gemini não encontrada. Erro: {e}")
    
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
        "2. Sempre cite as evidências correspondentes (ex: citando a Evidência [X], assim como os metadados relevantes no contexto da pergunta) para sustentar seus argumentos.\n"
        "3. Caso as evidências recuperadas sejam insuficientes, inconclusivas ou a pergunta esteja totalmente fora do escopo "
        "da base de conhecimento, responda obrigatoriamente: "
        "'Não possuo informações suficientes nas avaliações de clientes para responder a esta pergunta com segurança.'\n"
        "Instruções adicionais:\n"
        "Considere de baixo valor compras até 50, e compras de alto valor acima de 500."
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
