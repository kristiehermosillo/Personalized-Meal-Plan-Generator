# server.py
import os
from dotenv import load_dotenv  # <-- ADD THIS

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import stripe

load_dotenv()  # <-- ADD THIS to load variables from .env


# Env vars required:
# STRIPE_SECRET_KEY = "sk_live_..." or "sk_test_..."
# PRICE_ID = "price_..." (recurring or one-time)
# SUCCESS_URL = "http://localhost:8501/?session_id={CHECKOUT_SESSION_ID}"
# CANCEL_URL = "http://localhost:8501/"

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
PRICE_ID = os.getenv("PRICE_ID")
SUCCESS_URL = os.getenv("SUCCESS_URL", "http://127.0.0.1:8501/?session_id={CHECKOUT_SESSION_ID}")
CANCEL_URL = os.getenv("CANCEL_URL", "http://127.0.0.1:8501/")

if not STRIPE_SECRET_KEY:
    raise RuntimeError("Missing STRIPE_SECRET_KEY")
if not PRICE_ID:
    raise RuntimeError("Missing PRICE_ID")

stripe.api_key = STRIPE_SECRET_KEY

app = FastAPI(title="MealPlan Genie Backend")

# Allow Streamlit front-end to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CheckoutResponse(BaseModel):
    checkout_url: str

@app.post("/create-checkout-session", response_model=CheckoutResponse)
def create_checkout_session():
    try:
        session = stripe.checkout.Session.create(
            mode="payment",                # or "subscription" if PRICE_ID is recurring
            line_items=[{"price": PRICE_ID, "quantity": 1}],
            success_url=SUCCESS_URL,
            cancel_url=CANCEL_URL,
            allow_promotion_codes=True,
        )
        return CheckoutResponse(checkout_url=session.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/verify-session")
def verify_session(session_id: str):
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        paid = (session.get("payment_status") == "paid")
        return {"paid": bool(paid)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
