from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model (.json format)
model = xgb.XGBRegressor()
model.load_model("../model/dynamic_pricing_model.json")

# Define input schema
class ProductFeatures(BaseModel):
    category: int
    stock: int
    original_price: float
    competitor_price: float
    sales_rating: float
    last_week_sales: int
    discount_applied: float

# Health check route
@app.get("/")
def home():
    return {"message": "Dynamic Pricing API is live!"}

# Prediction route
@app.post("/predict")
def predict_price(data: ProductFeatures):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"recommended_price": round(prediction, 2)}
