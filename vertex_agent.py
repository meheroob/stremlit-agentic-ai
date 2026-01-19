
# Used to handle byte streams (CSV/PDF downloads)
import io

# Pandas is used to manipulate CSV data
import pandas as pd

# Google Cloud clients for Storage and BigQuery
from google.cloud import storage, bigquery

# Vertex AI SDK
import vertexai

# Used for text embeddings (vector search)
from vertexai.language_models import TextEmbeddingModel

# Used for generative responses (Gemini)
from vertexai.generative_models import GenerativeModel

# Streamlit session cache (used for RAG caching)
import streamlit as st


# -----------------------------
# Vertex AI Init
# -----------------------------

def init_vertex_ai(project_id: str, region: str):
    # Initialize Vertex AI with project and region
    vertexai.init(project=project_id, location=region)

    # Return Gemini Pro model for generation
    return GenerativeModel("gemini-2.5-pro")


# -----------------------------
# Load CSV from GCS
# -----------------------------

def load_user_data(bucket_name: str, blob_name: str):
    # Create Google Cloud Storage client
    client = storage.Client()

    # Reference the bucket
    bucket = client.bucket(bucket_name)

    # Reference the file inside the bucket
    blob = bucket.blob(blob_name)

    # Download CSV file as bytes
    data = blob.download_as_bytes()

    # Load CSV into a pandas DataFrame
    df = pd.read_csv(io.BytesIO(data), dtype=str)

    # Clean column names (remove spaces)
    df.columns = df.columns.str.strip()

    # Ensure CustomerID column exists
    if "CustomerID" not in df.columns:
        if "customerID" in df.columns:
            df.rename(columns={"customerID": "CustomerID"}, inplace=True)
        else:
            raise ValueError(
                f"'CustomerID' column not found in {blob_name}. Found: {df.columns.tolist()}"
            )

    # Set CustomerID as index for fast lookup
    df.set_index("CustomerID", inplace=True)

    return df


# -----------------------------
# Intent Classifier (Agentic Router)
# -----------------------------

def classify_domain(model, query: str) -> str:
    # Prompt instructs LLM to act as an intent router
    prompt = f"""
    Classify the following query into ONE category:
    - Pensions
    - Insurance
    - None

    Query: "{query}"

    Respond with exactly one word: Pensions, Insurance, or None.
    """

    try:
        # Generate classification
        response = model.generate_content(prompt)
        label = response.text.strip()

        # Validate output
        return label if label in ["Pensions", "Insurance", "None"] else "None"
    except Exception:
        # Safe fallback
        return "None"


# -----------------------------
# Standard Response Generator (Insurance)
# -----------------------------

def generate_response(model, customer_context: dict, query: str) -> str:
    # Convert customer context dictionary to readable text
    context_str = "\n".join(f"{k}: {v}" for k, v in customer_context.items())

    # Prompt for insurance queries (no RAG)
    prompt = f"""
Customer Context:
{context_str}

User Question:
{query}

Provide a clear, compliant response.
Always end with this disclaimer:
'DISCLAIMER: I am an AI model; please consult a financial advisor.'
    """

    # Generate response using Gemini
    response = model.generate_content(prompt)
    return response.text


# -----------------------------
# BigQuery RAG Setup (Pensions)
# -----------------------------

# BigQuery configuration
BQ_PROJECT = "project-2eaa6562-c344-4121-aad"
BQ_DATASET = "pdf_embeddings"
BQ_TABLE = "pension_chunks"

# Initialize Vertex AI for embeddings
vertexai.init(project=BQ_PROJECT, location="europe-central2")

# Load embedding model
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# Create BigQuery client
bq_client = bigquery.Client(project=BQ_PROJECT)


def retrieve_pension_chunks(query: str, top_k: int = 5) -> list:
    """
    Perform vector similarity search in BigQuery using dot product.
    """

    # Generate embedding for user query
    query_embedding = embedding_model.get_embeddings([query])[0].values

    # Convert embedding into SQL-friendly array
    query_str = ",".join(map(str, query_embedding))

    # SQL query computes dot product similarity between vectors
    sql = f"""
    SELECT datapoint_id, chunk_text,
           (SELECT SUM(x*y) 
            FROM UNNEST(embedding) AS x WITH OFFSET i
            JOIN UNNEST([{query_str}]) AS y WITH OFFSET j
            ON i=j) AS similarity
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    ORDER BY similarity DESC
    LIMIT {top_k}
    """

    # Execute query
    results = bq_client.query(sql).result()

    # Return top chunks (hidden from user)
    return [{"datapoint_id": r.datapoint_id, "chunk_text": r.chunk_text} for r in results]


# -----------------------------
# Cached Retrieval (Session-level)
# -----------------------------

def get_cached_pension_chunks(query: str, top_k: int = 5):
    # Initialize cache if missing
    if "pension_chunks_cache" not in st.session_state:
        st.session_state.pension_chunks_cache = {}

    cache_key = query.lower().strip()

    # Return cached result if exists
    if cache_key in st.session_state.pension_chunks_cache:
        return st.session_state.pension_chunks_cache[cache_key]

    # Otherwise query BigQuery
    chunks = retrieve_pension_chunks(query, top_k=top_k)
    st.session_state.pension_chunks_cache[cache_key] = chunks
    return chunks


# -----------------------------
# Pensions Response Generator (RAG)
# -----------------------------

def generate_pension_response(model, customer_context: dict, query: str) -> str:
    # Convert pension customer data into text
    context_str = "\n".join(f"{k}: {v}" for k, v in customer_context.items())

    # Retrieve relevant FCA chunks (internal only)
    chunks = get_cached_pension_chunks(query, top_k=5)
    combined_fca_context = "\n".join([c["chunk_text"] for c in chunks])

    # Prompt enforces compliance and hides internal context
    prompt = f"""
You are an AI assistant for a pensions customer.
Use the following context to answer the user's question with top 3 details from fca reference data.

Customer Context:
{context_str}

Internal FCA Reference Data (for your use only):
{combined_fca_context}

User Question:
{query}

Guidelines for your response:
- Provide accurate, compliant information about pensions.
- Include general knowledge like tax-free allowances, contribution limits, and standard rules where relevant.
- Do NOT give personalized advice or recommend specific actions.
- Keep it informative and educational.
- End with this disclaimer:
'DISCLAIMER: I am an AI model; this response is informed by FCA Handbook PS25/22; for personal advice, please consult a financial advisor.'
    """

    # Generate final response
    response = model.generate_content(prompt)
    return response.text
