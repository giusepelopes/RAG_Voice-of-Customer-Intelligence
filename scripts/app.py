# !pip install streamlit

import streamlit as st
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever import get_query_reply


st.set_page_config(page_title="RAG E-commerce - Voice of Customer", layout="wide")

st.title("Assistente de Inteligência de Negócios (Voice of Customer)")
st.markdown("Faça perguntas analíticas sobre as avaliações de clientes e obtenha insights a partir da base de conhecimento.")

n_results = st.slider(
    "Avaliações recuperadas:", 
    min_value=15, 
    max_value=50, 
    value=30, 
    step=5,
    help="Quantidade de evidências iniciais buscadas na base.",
    width=200
)

user_query = st.text_input("Qual é a sua pergunta?", placeholder="Ex: Quais são os principais problemas relatados pelos clientes?")

if st.button("Executar Consulta", type="primary"):
    if not user_query.strip():
        st.warning("Por favor, digite uma pergunta válida antes de consultar.")
    else:
        with st.spinner("Processando intenção, buscando evidências e gerando resposta..."):
            try:
                resposta, fontes = get_query_reply(user_query, n_results=n_results)
            except Exception as e:
                st.error(f"Ocorreu um erro ao processar a consulta: {e}")
                resposta, fontes = None, []

        if resposta:
            st.markdown("---")
            st.markdown("### Resposta do Assistente")
            st.info(resposta)

            if fontes:
                st.markdown("### Evidências Utilizadas")
                
                for fonte in fontes:
                    with st.expander(
                        f"Evidência [{fonte['evidencia_id']}] — Pedido: {fonte['order_id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**ID do Pedido:** `{fonte['order_id']}`")
                            st.write(f"**Nota da Avaliação:** {fonte['review_score']} estrelas")
                            st.write(f"**Estado do Cliente:** {fonte['customer_state']}")
                        with col2:
                            st.write(f"**Valor da Compra:** R$ {fonte['payment_value']:.2f}")
                            st.write(f"**Data da Compra:** {fonte.get('purchase_date', 'N/A')}")