# !pip install emoji

import emoji, re
import pandas as pd

def clean_review_text(text: str) -> str:
    if pd.isna(text):
        return ""

    # Remove emojis das reviews.
    text_no_emoji = emoji.replace_emoji(text, replace='')

    # Remove espaços extras ou quebras de linha indesejadas.
    clean_text = re.sub(r'\s+', ' ', text_no_emoji).strip()

    return clean_text

def create_review_document(row: pd.Series) -> str:
    # Criação do texto formatado de forma rica para que o embedding capture melhor o contexto de cada review.
    text = (f"[Avaliação do Cliente] "
            f"Nota: {row['review_score']} de 5 estrelas. | "
            f"Data da Compra: {row['order_purchase_timestamp']} | "
            f"Data Prevista para Entrega: {row['order_estimated_delivery_date']} | "
            f"Data de Entrega: {row['order_delivered_customer_date']} | "
            f"Valor da compra: R$ {row['payment_value']} | "
            f"Comentário: {row['clean_message']}")
    return text

def get_treated_data() -> pd.DataFrame:
    print(f"[DTREAT:LOG] Iniciando tratamento de dados.")

    try:
        df_reviews = pd.read_csv('../data/olist_order_reviews_dataset.csv')
        df_orders = pd.read_csv('../data/olist_orders_dataset.csv')
        df_payments = pd.read_csv('../data/olist_order_payments_dataset.csv')
    except Exception as e:
        print(f"[DTREAT:ERR] Falha na leitura dos arquivos de entrada: {str(e)}")
        raise

    """Todas as reviews possuem score. Score na faixa de 1 a 5."""

    print("[DTREAT:LOG] Total de avaliações original:", len(df_reviews))

    df_merged_orders = pd.merge(df_reviews, df_orders[['order_id', 'order_purchase_timestamp', 'order_estimated_delivery_date', 'order_delivered_customer_date']], on='order_id', how='inner')

    # Aplica função de limpeza ao texto no campo de mensagem da review.
    df_merged_orders['clean_message'] = df_merged_orders['review_comment_message'].apply(clean_review_text)

    # Remove avaliações vazias ou com frases que trazem pouco contexto útil, como "ok" ou "bom".
    df_merged_orders = df_merged_orders[df_merged_orders['clean_message'].str.len() > 3].copy()

    print("[DTREAT:LOG] Total de avaliações após limpeza:", len(df_merged_orders))

    df_payments = df_payments[['order_id', 'payment_value']].groupby('order_id').agg('sum').reset_index()
    df_pay_enrich = pd.merge(df_merged_orders, df_payments, on='order_id', how='left')

    df_merged = df_pay_enrich

    # Aplicar a formatação criando uma coluna de texto estruturado
    df_merged['rag_document'] = df_merged.apply(create_review_document, axis=1)

    print(f"[DTREAT:LOG] Tratamento de dados concluído com sucesso.")
    
    return df_merged