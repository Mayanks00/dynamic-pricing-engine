from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

app = FastAPI()

# ✅ Load updated model
model = xgb.XGBRegressor()
model.load_model("model/dynamic_pricing_model.json")

# ✅ UPDATED category mapping (must match model training)
category_map = {
    "electronics": 0,
    "clothing": 1,
    "beauty": 2,
    "home": 3,
    "sports": 4,
    "furniture": 5,
    "toys": 6,
    "books": 7
}

# ✅ Input schema
class ProductFeatures(BaseModel):
    product_id: int
    category: str
    stock: int
    original_price: float
    competitor_price: float
    sales_rating: float
    last_week_sales: int
    discount_applied: float

@app.get("/")
def home():
    return {"message": "Dynamic Pricing API is live!"}

@app.post("/predict_price")
def predict_price(data: ProductFeatures):
    try:
        input_data = data.dict()

        # 🔁 Case-insensitive category handling
        category_str = input_data.get("category", "").strip().lower()

        if category_str not in category_map:
            return {"error": f"Invalid category: '{category_str}'. Must be one of {list(category_map.keys())}"}

        input_data["category"] = category_map[category_str]

        # Remove product_id (not used in model)
        input_data.pop("product_id", None)

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]

        return {
            "recommended_price": round(float(prediction), 2),
            "input_received": input_data  # ➕ Show inputs used for prediction (optional)
        }

    except Exception as e:
        return {"error": "Prediction Failed", "details": str(e)}
