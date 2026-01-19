# Streamlit is used to build the web UI
import streamlit as st

# Import agent logic from vertex_agent.py
from vertex_agent import (
    init_vertex_ai,             # initializes Gemini model
    load_user_data,             # loads CSVs from GCS
    classify_domain,            # classifies query intent
    generate_response,          # insurance response generator
    generate_pension_response,  # RAG-based pension response generator
)

# -----------------------------
# Config
# -----------------------------

# GCP project ID used for Vertex AI and BigQuery
PROJECT_ID = "project-2eaa6562-c344-4121-aad"

# Region where Vertex AI models are hosted
REGION = "europe-central2"

# GCS bucket containing user CSV files
BUCKET_NAME = "meheroob-chatbot-dummy-data"

# Filenames inside the bucket
ALL_USERS_FILE = "all-users.csv"
PENSIONS_FILE = "users-pensions.csv"
INSURANCE_FILE = "user-insurance.csv"

# -----------------------------
# Init Model
# -----------------------------

# Initialize Vertex AI and load the Gemini model once at startup
model = init_vertex_ai(PROJECT_ID, REGION)

# -----------------------------
# Load Data (cached)
# -----------------------------

# Cache CSV loading so Streamlit doesn't re-download data on every rerun
@st.cache_data
def load_all_data():
    return (
        load_user_data(BUCKET_NAME, ALL_USERS_FILE),
        load_user_data(BUCKET_NAME, PENSIONS_FILE),
        load_user_data(BUCKET_NAME, INSURANCE_FILE),
    )

# Load all user-related datasets
all_users_df, pensions_df, insurance_df = load_all_data()

# -----------------------------
# Session State
# -----------------------------

# Store authenticated customer in Streamlit session
if "customer" not in st.session_state:
    st.session_state.customer = None

# App title
st.title("Agentic Insurance & Pension Assistant")

# -----------------------------
# AUTHENTICATION
# -----------------------------

# If no customer is logged in yet
if st.session_state.customer is None:

    # Ask user to input their CustomerID
    customer_id = st.text_input("Enter your CustomerID")

    if customer_id:
        customer_id = customer_id.strip()

        # Validate CustomerID
        if customer_id not in all_users_df.index:
            st.warning("CustomerID not found. Please try again.")
        else:
            """
            The idea is to authenticate and classify based on the fact that does a customer hold pension or investment
            product or both. It's done by 2 booleans has_pension and has_insurance
            """
            # Retrieve user identity data
            identity = all_users_df.loc[customer_id].to_dict()

            # Check which products the user owns
            has_pension = customer_id in pensions_df.index
            has_insurance = customer_id in insurance_df.index

            # Store customer profile in session state
            st.session_state.customer = {
                "id": customer_id,
                "identity": identity,
                "has_pension": has_pension,
                "has_insurance": has_insurance,
                "pension": pensions_df.loc[customer_id].to_dict() if has_pension else None,
                "insurance": insurance_df.loc[customer_id].to_dict() if has_insurance else None,
            }

            full_name = f"{identity['FirstName']} {identity['LastName']}"

            # Personalized greeting depending on products owned
            if has_pension and has_insurance:
                st.success(
                    f"Hi {full_name}, how can I help you with your insurance and pension queries?"
                )
            elif has_pension:
                st.success(
                    f"Hi {full_name}, how can I help you with your pension queries?"
                )
            elif has_insurance:
                st.success(
                    f"Hi {full_name}, how can I help you with your insurance queries?"
                )
            else:
                st.warning("No active products found for your account.")

# -----------------------------
# CHAT LOOP
# -----------------------------

# Only enable chat once the user is authenticated
if st.session_state.customer:

    # Chat input box
    if prompt := st.chat_input("Ask your question"):

        # Display user message
        st.chat_message("user").markdown(prompt)

        # Classify intent using LLM (agent router)
        domain = classify_domain(model, prompt)
        customer = st.session_state.customer

        # Route to pensions logic
        if domain == "Pensions":
            if not customer["has_pension"]:
                answer = "Unfortunately, you do not have a pension account with us."
            else:
                # RAG-informed response using FCA documents
                answer = generate_pension_response(
                    model, customer["pension"], prompt
                )

        # Route to insurance logic
        elif domain == "Insurance":
            if not customer["has_insurance"]:
                answer = "Unfortunately, you do not have an insurance policy with us."
            else:
                # Standard LLM response using customer policy data
                answer = generate_response(
                    model, customer["insurance"], prompt
                )

        # Fallback for chit-chat or unrelated queries
        else:
            answer = (
                "I'm happy to chat! I can help with pension or insurance questions anytime."
            )

        # Display assistant response
        st.chat_message("assistant").markdown(answer)
