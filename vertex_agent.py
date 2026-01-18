import io
import pandas as pd
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

# -----------------------------
# Vertex AI Init
# -----------------------------
def init_vertex_ai(project_id: str, region: str):
    vertexai.init(project=project_id, location=region)
    return GenerativeModel("gemini-2.5-pro")


# -----------------------------
# Load CSV from GCS
# -----------------------------
def load_user_data(bucket_name: str, blob_name: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    data = blob.download_as_bytes()
    df = pd.read_csv(io.BytesIO(data), dtype=str)

    # Normalize column names
    df.columns = df.columns.str.strip()

    # Handle inconsistent CustomerID naming
    if "CustomerID" not in df.columns:
        if "customerID" in df.columns:
            df.rename(columns={"customerID": "CustomerID"}, inplace=True)
        else:
            raise ValueError(
                f"'CustomerID' column not found in {blob_name}. Found: {df.columns.tolist()}"
            )

    df.set_index("CustomerID", inplace=True)
    return df



# -----------------------------
# Intent Classifier (Agentic Router)
# -----------------------------
def classify_domain(model, query: str) -> str:
    prompt = f"""
    Classify the following query into ONE category:
    - Pensions
    - Insurance
    - None

    Query: "{query}"

    Respond with exactly one word: Pensions, Insurance, or None.
    """

    try:
        response = model.generate_content(prompt)
        label = response.text.strip()
        return label if label in ["Pensions", "Insurance", "None"] else "None"
    except Exception:
        return "None"


# -----------------------------
# RAG Response Generator
# -----------------------------
def generate_response(model, customer_context: dict, query: str) -> str:
    context_str = "\n".join(f"{k}: {v}" for k, v in customer_context.items())

    prompt = f"""
    Customer Context:
    {context_str}

    User Question:
    {query}

    Provide a clear, compliant response.
    Always end with this disclaimer:
    'DISCLAIMER: I am an AI model; please consult a financial advisor.'
    """

    response = model.generate_content(prompt)
    return response.text
