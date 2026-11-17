# Assistente de Inteligência de Negócios — RAG (Voice of Customer)

Este projeto implementa um pipeline **RAG (Retrieval-Augmented Generation)**  voltado para a análise de avaliações de clientes de e-commerce (*Voice of Customer*). O sistema processa grandes volumes de dados de avaliações, realiza buscas semânticas precisas combinadas com filtros estruturados e utiliza Inteligência Artificial para gerar insights analíticos acionáveis para gestores.


## Arquitetura do Pipeline

O fluxo de funcionamento do assistente é dividido em 6 etapas principais:

### 1. Ingestão e Tratamento de Dados (`data_treatment.py`)
* **Unificação de Bases:** Leitura e cruzamento de bases relacionais de pedidos, avaliações (*reviews*), pagamentos e dados dos clientes.
* **Limpeza de Texto:** Remoção automática de emojis e normalização de quebras de linha e espaços em branco nas mensagens.
* **Filtragem de Ruído:** Descarte de avaliações vazias ou com frases muito curtas (≤ 3 caracteres) que agregam pouco valor analítico.
* **Enriquecimento:** Consolidação do valor total de pagamentos por pedido e formatação de um documento textual rico (`rag_document`) combinando metadados e o comentário do cliente.

### 2. Indexação e Armazenamento Vetorial (`indexing.py`)
* **Armazenamento:** Inicialização de um cliente persistente utilizando o **ChromaDB**.
* **Geração de Embeddings:** Conversão dos textos estruturados em vetores utilizando o modelo **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) via *Sentence Transformers*.
* **Persistência em Lotes:** Inserção otimizada em *batches* dos documentos vetorizados acompanhados de metadados estruturados (nota, ID do pedido, valor da compra, estado e data).

### 3. Interface do Usuário (`app.py`)
* **Aplicação Web:** Interface interativa e amigável desenvolvida em **Streamlit**.
* **Parâmetros Dinâmicos:** Controle deslizante para o usuário definir a quantidade de evidências iniciais recuperadas na base.
* **Auditoria de Resposta:** Exibição detalhada da resposta gerada pelo assistente junto com um painel expansível contendo as evidências originais utilizadas.

### 4. Extração de Intenção e Filtros Dinâmicos (`retriever.py`)
* **Classificação de Intenção:** Utilização da LLM (`gemini-3.5-flash-lite`) para analisar a pergunta do usuário e extrair restrições explícitas ou implícitas (como faixas de notas, valores de compras ou estados/regiões) estruturadas em JSON.
* **Filtros Inteligentes:** Tradução automática da intenção extraída em consultas estruturadas de metadados aplicadas diretamente no ChromaDB (`where=dynamic_filter`).

### 5. Recuperação e Reclassificação (*Retrieval & Reranking*) (`retriever.py`)
* **Busca Semântica:** Recuperação inicial dos candidatos mais relevantes na base vetorial com base na consulta filtrada.
* **Reranking:** Aplicação de um modelo *Cross-Encoder* (`BAAI/bge-reranker-base`) para reclassificar rigorosamente a relevância dos pares entre a pergunta e os documentos recuperados, isolando apenas o *top* mais assertivo.

### 6. Geração de Resposta Baseada em Evidências (`retriever.py`)
* **Engenharia de Prompt:** Construção de um prompt especializado definindo o papel de assistente sênior de BI, com diretrizes rígidas contra alucinações e obrigatoriedade de citação das fontes.
* **Geração Resiliente:** Acionamento da LLM (`gemini-3.5-flash`) integrada com um mecanismo de tentativas (*retries*) para garantir estabilidade e resiliência frente a limites de taxa da API.


## Tecnologias Utilizadas

* **Python**
* **Streamlit** (Interface Gráfica)
* **Google GenAI SDK** (LLM e Classificação de Intenções)
* **ChromaDB** (Banco de Dados Vetorial)
* **Sentence Transformers & BERTimbau** (`neuralmind/bert-base-portuguese-cased`) (Embeddings)
* **Cross-Encoder** (`BAAI/bge-reranker-base`) (Reclassificação de Evidências)
* **Pandas** (Manipulação e Tratamento de Dados)

## Execução do Projeto

### Local

* Instale as bibliotecas necessárias para execução do projeto.

        cd RAG_Voice-of-Customer-Intelligence/
        pip install -r requirements.txt

* Navegue para a pasta de scripts e execute "indexing.py". **Nota: esse passo deve levar alguns minutos para conclusão, visto que irá tratar os dados e persisti-los no banco vetorial (>50k registros).**
    
        cd scripts/
        python indexing.py

* Exporte o valor da sua chave para API Gemini para variável de ambiente ([Documentação Google AI Studio](https://aistudio.google.com/docs/api-key)).

        export GEMINI_API_KEY="SUA-CHAVE"

* Execute a aplicação. Serão apresentados os IPs que poderão ser utilizados para acesso via **porta 8501**.

        streamlit run app.py

### Via Docker

* Navegue para a pasta de scripts e execute "indexing.py".
    
        cd RAG_Voice-of-Customer-Intelligence/scripts/
        python indexing.py

* Navegue para raiz do projeto e construa a imagem Docker.

        cd ..
        docker build -t rag-ecommerce-app .

* Execute a aplicação. Também serão apresentados os IPs que poderão ser utilizados para acesso via **porta 8501**.

        docker run -p 8501:8501 -e GEMINI_API_KEY="SUA-CHAVE" rag-ecommerce-app
