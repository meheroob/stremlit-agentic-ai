import io
import math
import pdfplumber # PDF text extraction

# GCP clients
from google.cloud import storage, bigquery

# Vertex AI SDK
import vertexai
from vertexai.language_models import TextEmbeddingModel


# GCS source configuration
BUCKET_NAME = "meheroob-chatbot-dummy-data"
PREFIX = "my-fca-docs/pensions/"

# BigQuery destination
PROJECT_ID = "project-2eaa6562-c344-4121-aad"
DATASET_ID = "pdf_embeddings"
TABLE_ID = "pension_chunks"


# -----------------------------
# Load PDFs from GCS
# -----------------------------

# Create GCS client
client = storage.Client()

# List all files under pensions folder
blobs = client.list_blobs(BUCKET_NAME, prefix=PREFIX)

texts = []

# Iterate through PDFs 
for blob in blobs:
    if not blob.name.endswith(".pdf"):
        continue

    # Download PDF into memory
    pdf_bytes = blob.download_as_bytes()

    # Extract text page by page
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text)

# Combine all extracted text
full_text = "\n".join(texts)
len(full_text)


# -----------------------------
# Chunking
# -----------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    # Split text into words
    words = text.split()
    chunks = []
    start = 0

    # Sliding window chunking
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# Create text chunks
chunks = chunk_text(full_text)


# -----------------------------
# Embeddings
# -----------------------------

# Initialize Vertex AI
vertexai.init(
    project=PROJECT_ID,
    location="europe-central2"
)

# Load embedding model
embedding_model = TextEmbeddingModel.from_pretrained(
    "text-embedding-004"
)


def embed_in_batches(texts, batch_size=20):
    # Store all embeddings
    all_embeddings = []

    # Generate embeddings in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = embedding_model.get_embeddings(batch)
        all_embeddings.extend([e.values for e in response])

    return all_embeddings


# Generate embeddings
embeddings = embed_in_batches(chunks)


# -----------------------------
# BigQuery Storage
# -----------------------------

# Create BigQuery client
bq_client = bigquery.Client(project=PROJECT_ID)

# Fully qualified table name
table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"

# Create dataset if not exists
dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
dataset.location = "europe-central2"
bq_client.create_dataset(dataset, exists_ok=True)

# Define schema
schema = [
    bigquery.SchemaField("datapoint_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("chunk_text", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
]

# Create table if missing
table = bigquery.Table(table_ref, schema=schema)
bq_client.create_table(table, exists_ok=True)

# Insert data in batches as there is a limit
BATCH_SIZE = 500
num_batches = math.ceil(len(embeddings) / BATCH_SIZE)

for i in range(0, len(embeddings), BATCH_SIZE):
    batch_embeddings = embeddings[i:i + BATCH_SIZE]
    batch_chunks = chunks[i:i + BATCH_SIZE]

    rows_to_insert = [
        {
            "datapoint_id": str(i + j),
            "chunk_text": batch_chunks[j],
            "embedding": batch_embeddings[j],
        }
        for j in range(len(batch_embeddings))
    ]

    errors = bq_client.insert_rows_json(table_ref, rows_to_insert)

    if errors:
        print(f"Errors in batch {i}-{i + len(batch_embeddings)}:", errors)
    else:
        print(f"Inserted batch {i}-{i + len(batch_embeddings)} successfully")