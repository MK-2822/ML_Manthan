# FreshEye AI 🍎🍌🍊

FreshEye AI is an intelligent fruit quality classification dashboard that analyzes images of fruits (apples, bananas, oranges) to determine their freshness. It utilizes a hybrid AI architecture, leveraging Google's Gemini Vision API for high-accuracy analysis, with a seamless fallback to a local custom-trained TensorFlow/Keras CNN model.

## 🌟 Key Features
- **Validation AI Architecture**: Reverses the typical API logic! We query your custom-trained **TensorFlow CNN (`model1.h5`)** *first*. Then, we pass those initial test scores to **Google Gemini 2.5 Flash**, asking it to visually *validate* and correct the CNN's assumptions, thereby significantly enhancing your model's real-world accuracy.
- **Local Fallback Built-In**: If you are offline, or the API is unavailable/unconfigured, the system automatically accepts the Base CNN's predictions.
- **Modern Glassmorphism UI**: Beautiful, fully responsive web interface built with Tailwind CSS and Vanilla JavaScript.
- **Smart Recommendations**: Provides dynamic, context-aware advice on shelf life, storage methods, and prime usage (e.g., baking vs. fresh eating) based on the calculated freshness percentage.
- **Scan History**: Keeps track of your recently analyzed fruits during your session.
- **Dark Mode**: Fully supports browser-level or manual dark mode toggling.

## 🛠️ Tech Stack
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS (via CDN), FontAwesome.
- **Backend**: FastAPI (Python), Uvicorn.
- **Machine Learning**: 
  - `google-genai` (Gemini API)
  - `tensorflow` & `keras` (Local CNN for Fresh vs. Rotten classification)
- **Image Processing**: `Pillow` (PIL), `numpy`

## 📂 Project Structure
```text
ManthonML/
│
├── static/                     # Frontend assets (HTML, CSS, JS)
│   └── index.html              # Main Single Page Application UI
├── model1.h5                   # Custom-trained local TensorFlow CNN model
├── backend.py                  # FastAPI server & AI logic pipeline
├── Fruit_Quality_Classification.ipynb # Original ML training notebook
├── .env                        # Environment variables (API Keys)
└── README.md                   # Project documentation
```

## 🚀 Installation & Setup

1. **Navigate to the project directory**
   ```bash
   cd ManthonML
   ```

2. **Create and activate a virtual environment**
   - **Windows:**
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   Make sure you have the required packages installed:
   ```bash
   pip install fastapi uvicorn python-multipart tensorflow numpy pillow python-dotenv google-genai
   ```

4. **Environment Variables**
   Ensure there is a `.env` file in the root directory (same level as `backend.py`). Add your Google AI Studio API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
   *(If you leave it blank or as the placeholder, the app will automatically route all requests to the local TensorFlow fallback model).*

5. **Start the Server**
   Run the FastAPI server using Uvicorn or Python directly:
   ```bash
   python backend.py
   ```
   *or*
   ```bash
   uvicorn backend:app --reload
   ```

6. **Access the Web App**
   Open your browser and navigate to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 📸 How to Use
1. Open the web interface.
2. Click the upload area or **drag and drop** an image of an apple, banana, or orange.
3. The AI will process the image (attempting Gemini first, then local ML).
4. View the result: Freshness score, specific item detected, and tailored storage recommendations!

## ⚙️ Model Details
The local CNN (`model1.h5`) was trained on 6 specific classes. During fallback execution, the FastAPI backend dynamically calculates probabilities across these classes and aggregates them into "Fresh" vs. "Rotten" confidence scores:
- Fresh Apple, Fresh Banana, Fresh Orange
- Rotten Apple, Rotten Banana, Rotten Orange