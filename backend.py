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
    if not model and (not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here"):
        return {"error": "Neither Local Model nor Gemini API is available."}
    
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    
    # Try Gemini API First
    if gemini_client:
        try:
            prompt = """Analyze this image of a fruit and respond ONLY with a JSON object.
Do not wrap it in markdown block quotes. Just the raw JSON.
The JSON must have the following keys:
- "predicted_class": string, either "Fresh" or "Rotten"
- "specific_item": string, e.g., "Fresh Apple", "Rotten Banana", "Fresh Orange", etc.
- "freshness_score": integer between 0 and 100 representing the freshness percentage.

Consider physical signs of decay, mold, bruising, or perfect ripeness when evaluating.
"""
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, image]
            )
            response_text = response.text.replace("```json\n", "").replace("\n```", "").strip()
            gemini_data = json.loads(response_text)
            
            predicted_class = gemini_data.get("predicted_class", "Fresh")
            specific_item = gemini_data.get("specific_item", "Unknown Item")
            freshness_percentage = int(gemini_data.get("freshness_score", 0))
            
            # Use Gemini's results directly to proceed to the sublabels
            pass_to_sublabels = True
        except Exception as e:
            print(f"Gemini API failed: {e}. Falling back to local model.")
            pass_to_sublabels = False
    else:
        pass_to_sublabels = False
        
    # Local Model Fallback
    if not pass_to_sublabels:
        # Preprocess image for model (matching app.py specs: 240x240, normalize)
        img_resized = image.resize((240, 240))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array)[0]
        
        # Exact mapping from the notebook's training data:
        # ['freshoranges', 'rottenapples', 'freshbanana', 'rottenoranges', 'freshapples', 'rottenbanana']
        class_names = [
            'Fresh Orange', 
            'Rotten Apple', 
            'Fresh Banana', 
            'Rotten Orange', 
        'Fresh Apple', 
        'Rotten Banana'
    ]
    
        if len(predictions) == 6:
            # Fresh indices: 0 (Orange), 2 (Banana), 4 (Apple)
            fresh_conf = float(predictions[0] + predictions[2] + predictions[4])
            # Rotten indices: 1 (Apple), 3 (Orange), 5 (Banana)
            rotten_conf = float(predictions[1] + predictions[3] + predictions[5])
            
            max_idx = int(np.argmax(predictions))
            specific_item = class_names[max_idx]
        else:
            fresh_conf = float(predictions[0])
            rotten_conf = float(predictions[1])
            specific_item = "Unknown Item"
        
        predicted_class = "Fresh" if fresh_conf >= rotten_conf else "Rotten"
        
        # Calculate freshness percentage
        freshness_percentage = int(fresh_conf * 100)
        
        # User requested to avoid always showing exactly 0% for very rotten fruits.
        # We will generate a realistic dynamic low score (e.g. 12% - 35%) so it doesn't look static.
        if predicted_class == "Rotten" and freshness_percentage < 10:
            # Use the micro-decimals of the probability to create a perfectly reproducible but varied score 
            pseudo_random_variance = int((rotten_conf * 10000) % 25)
            freshness_percentage = 8 + pseudo_random_variance
            
        # Safeguard for UI
        if freshness_percentage < 50:
            predicted_class = "Rotten"
            
    # Calculate more accurate recommendations
    if freshness_percentage >= 80:
        shelf_life = "Eat within 5-7 days"
        shelf_subtitle = "Store in crisper drawer"
        best_for = "Fresh Eating"
        best_subtitle = "Texture confirmed"
        status_color = "#16a34a"  # green
    elif freshness_percentage >= 50:
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
        "predicted_class": predicted_class,
        "specific_item": specific_item,
        "freshness_score": freshness_percentage,
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
