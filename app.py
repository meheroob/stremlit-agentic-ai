import streamlit as st
from vertex_agent import (
    init_vertex_ai,
    load_user_data,
    classify_domain,
    generate_response,
)

# -----------------------------
# Config
# -----------------------------
PROJECT_ID = "project-2eaa6562-c344-4121-aad"
REGION = "europe-central2"
BUCKET_NAME = "meheroob-chatbot-dummy-data"

ALL_USERS_FILE = "all-users.csv"
PENSIONS_FILE = "users-pensions.csv"
INSURANCE_FILE = "user-insurance.csv"

# -----------------------------
# Init Model
# -----------------------------
model = init_vertex_ai(PROJECT_ID, REGION)

# -----------------------------
# Load Data (cached)
# -----------------------------
@st.cache_data
def load_all_data():
    return (
        load_user_data(BUCKET_NAME, ALL_USERS_FILE),
        load_user_data(BUCKET_NAME, PENSIONS_FILE),
        load_user_data(BUCKET_NAME, INSURANCE_FILE),
    )

all_users_df, pensions_df, insurance_df = load_all_data()

# -----------------------------
# Session State
# -----------------------------
if "customer" not in st.session_state:
    st.session_state.customer = None

st.title("Agentic Insurance & Pension Assistant")

# -----------------------------
# AUTHENTICATION
# -----------------------------
if st.session_state.customer is None:
    customer_id = st.text_input("Enter your CustomerID")

    if customer_id:
        customer_id = customer_id.strip()

        if customer_id not in all_users_df.index:
            st.warning("CustomerID not found. Please try again.")
        else:
            identity = all_users_df.loc[customer_id].to_dict()

            has_pension = customer_id in pensions_df.index
            has_insurance = customer_id in insurance_df.index

            st.session_state.customer = {
                "id": customer_id,
                "identity": identity,
                "has_pension": has_pension,
                "has_insurance": has_insurance,
                "pension": pensions_df.loc[customer_id].to_dict() if has_pension else None,
                "insurance": insurance_df.loc[customer_id].to_dict() if has_insurance else None,
            }

            full_name = f"{identity['FirstName']} {identity['LastName']}"

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
if st.session_state.customer:
    if prompt := st.chat_input("Ask your question"):
        st.chat_message("user").markdown(prompt)

        domain = classify_domain(model, prompt)
        customer = st.session_state.customer

        if domain == "Pensions":
            if not customer["has_pension"]:
                answer = "Unfortunately, you do not have a pension account with us."
            else:
                answer = generate_response(
                    model, customer["pension"], prompt
                )

        elif domain == "Insurance":
            if not customer["has_insurance"]:
                answer = "Unfortunately, you do not have an insurance policy with us."
            else:
                answer = generate_response(
                    model, customer["insurance"], prompt
                )

        else:
            answer = (
                "I'm happy to chat! I can help with pension or insurance questions anytime."
            )

        st.chat_message("assistant").markdown(answer)
