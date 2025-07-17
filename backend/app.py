from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load trained XGBoost model
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

# Root route for health check
@app.get("/")
def home():
    return {"message": "Dynamic Pricing API is live!"}

# Prediction route
@app.post("/predict")
def predict_price(data: ProductFeatures):
    try:
        input_data = data.dict()

        # Convert category label to numeric
        category_str = input_data.get("category")
        if category_str not in category_map:
            return {"error": f"Invalid category: {category_str}. Must be one of {list(category_map.keys())}"}

        input_data["category"] = category_map[category_str]

        # Optional: remove product_id if model not trained on it
        input_data.pop("product_id", None)

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]

        return {"recommended_price": round(float(prediction), 2)}

    except Exception as e:
        print("🔥 Error during prediction:", e)
        return {"error": "Internal Server Error", "details": str(e)}


