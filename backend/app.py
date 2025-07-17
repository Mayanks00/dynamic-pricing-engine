from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = xgb.XGBRegressor()
model.load_model("model/dynamic_pricing_model.json")

# Category label to number mapping
category_map = {
    "electronics": 0,
    "clothing": 1,
    "beauty": 2,
    "home": 3
}

# Define input schema
class ProductFeatures(BaseModel):
    product_id: int
    category: str
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
    input_data = data.dict()

    # Convert category string to integer
    if input_data["category"] in category_map:
        input_data["category"] = category_map[input_data["category"]]
    else:
        return {"error": "Unknown category"}

    # Drop product_id (not used in model)
    input_data.pop("product_id")

    # Create DataFrame and predict
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    return {"recommended_price": round(prediction, 2)}

