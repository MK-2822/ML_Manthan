from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = tf.keras.models.load_model("model1.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model and not gemini_client:
        return {"error": "Neither Local Model nor Gemini API is available."}
    
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    
    local_predicted_class = "Unknown"
    local_specific_item = "Unknown Item"
    local_freshness_percentage = 0
    local_model_success = False
    
    # 1. Primary pass: Run Local Model First
    if model:
        try:
            # Preprocess image for model (matching app.py specs: 240x240, normalize)
            img_resized = image.resize((240, 240))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)[0]
            
            # Exact mapping from the notebook's training data:
            class_names = [
                'Fresh Orange', 'Rotten Apple', 'Fresh Banana', 
                'Rotten Orange', 'Fresh Apple', 'Rotten Banana'
            ]
        
            if len(predictions) == 6:
                fresh_conf = float(predictions[0] + predictions[2] + predictions[4])
                rotten_conf = float(predictions[1] + predictions[3] + predictions[5])
                
                max_idx = int(np.argmax(predictions))
                local_specific_item = class_names[max_idx]
            else:
                fresh_conf = float(predictions[0])
                rotten_conf = float(predictions[1])
                local_specific_item = "Unknown Item"
            
            local_predicted_class = "Fresh" if fresh_conf >= rotten_conf else "Rotten"
            local_freshness_percentage = int(fresh_conf * 100)
            
            # Dynamic low score calculation
            if local_predicted_class == "Rotten" and local_freshness_percentage < 10:
                pseudo_random_variance = int((rotten_conf * 10000) % 25)
                local_freshness_percentage = 8 + pseudo_random_variance
                
            if local_freshness_percentage < 50:
                local_predicted_class = "Rotten"
                
            local_model_success = True
        except Exception as e:
            print(f"Local model prediction failed: {e}")

    # Set base results
    final_predicted_class = local_predicted_class
    final_specific_item = local_specific_item
    final_freshness_percentage = local_freshness_percentage
    
    # 2. Secondary pass: Gemini API for Validation & Accuracy Enhancement
    if gemini_client:
        try:
            if local_model_success:
                prompt = f"""Our local CNN fruit model analyzed this image and predicted it as a "{local_specific_item}" with an overall status of "{local_predicted_class}" and a freshness score of {local_freshness_percentage}/100.
                
Please perform a visual analysis to validate or correct this prediction to increase the final accuracy. Look closely at the fruit's physical signs of decay, mold, bruising, or perfect ripeness. 

Respond ONLY with a JSON object. Do not wrap it in markdown block quotes. Just the raw JSON.
The JSON must have the following keys:
- "predicted_class": string, either "Fresh" or "Rotten"
- "specific_item": string, e.g., "Fresh Apple", "Rotten Banana", "Fresh Orange", etc.
- "freshness_score": integer between 0 and 100 representing the freshness percentage.
"""
            else:
                prompt = """Analyze this image of a fruit and respond ONLY with a JSON object.
Do not wrap it in markdown block quotes. Just the raw JSON.
The JSON must have the following keys:
- "predicted_class": string, either "Fresh" or "Rotten"
- "specific_item": string, e.g., "Fresh Apple", "Rotten Banana", "Fresh Orange", etc.
- "freshness_score": integer between 0 and 100 representing the freshness percentage.
"""
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, image]
            )
            response_text = response.text.replace("```json\n", "").replace("\n```", "").strip()
            gemini_data = json.loads(response_text)
            
            # Adopt Gemini's validated findings
            final_predicted_class = gemini_data.get("predicted_class", final_predicted_class)
            final_specific_item = gemini_data.get("specific_item", final_specific_item)
            final_freshness_percentage = int(gemini_data.get("freshness_score", final_freshness_percentage))
            
        except Exception as e:
            print(f"Gemini Validation failed: {e}. Falling back purely to local model results.")

    if not local_model_success and not gemini_client:
        return {"error": "Prediction pipeline failed. Both local model and Gemini API are unavailable."}

    # Calculate more accurate recommendations based on the final validated data
    if final_freshness_percentage >= 80:
        shelf_life = "Eat within 5-7 days"
        shelf_subtitle = "Store in crisper drawer"
        best_for = "Fresh Eating"
        best_subtitle = "Texture confirmed"
        status_color = "#16a34a"  # green
    elif final_freshness_percentage >= 50:
        shelf_life = "Consume soon"
        shelf_subtitle = "Keep refrigerated"
        best_for = "Baking / Smoothies"
        best_subtitle = "Slightly soft texture"
        status_color = "#eab308"  # yellow
    else:
        shelf_life = "Discard immediately"
        shelf_subtitle = "Do not store with fresh produce"
        best_for = "Composting"
        best_subtitle = "Not safe for consumption"
        status_color = "#dc2626"  # red
        
    return {
        "status": "success",
        "predicted_class": final_predicted_class,
        "specific_item": final_specific_item,
        "freshness_score": final_freshness_percentage,
        "details": {
            "shelf_life": shelf_life,
            "shelf_subtitle": shelf_subtitle,
            "best_for": best_for,
            "best_subtitle": best_subtitle,
            "status_color": status_color
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
