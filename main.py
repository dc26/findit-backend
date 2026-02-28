import os
import time
import json
import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yt_dlp
import google.generativeai as genai

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins so index.html works from anywhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class ReelRequest(BaseModel):
    url: str

class RestaurantLocation(BaseModel):
    name: str
    lat: float
    lng: float
    cuisine: str
    rating: str

# --- Helper Functions ---

def download_audio(url: str, output_path: str = "audio.mp3") -> str:
    """Downloads the audio from an Instagram Reel using yt-dlp."""
    logger.info(f"Downloading audio from {url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'audio.%(ext)s',
        'quiet': False, # set to false for debugging
    }

    try:
        # Ensure cleanup of old file
        if os.path.exists(output_path):
            os.remove(output_path)
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        if os.path.exists(output_path):
            return output_path
        else:
            raise Exception("File was not created by yt-dlp")
            
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        raise HTTPException(status_code=400, detail=f"Could not download reel audio. Ensure it is a valid public IG Reel URL. Error: {str(e)}")

def extract_restaurants_with_gemini(audio_path: str) -> list:
    """Uploads audio to Gemini and extracts a JSON list of restaurants."""
    if not GEMINI_API_KEY:
         raise HTTPException(status_code=500, detail="Gemini API Key is not configured on the server.")

    try:
        logger.info("uploading file to Gemini API...")
        uploaded_file = genai.upload_file(path=audio_path)
        
        logger.info("Initializing lowest-friction Gemini 1.5 Pro model...")
        
        # Dynamically find the correct 1.5 Pro string for this specific SDK version to prevent 404s
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        logger.info(f"Available models: {available_models}")
        
        target_model_name = 'models/gemini-1.5-pro' # default fallback
        
        for name in available_models:
             # Find standard 1.5 pro (not flash, not vision)
             if 'gemini-1.5-pro' in name and 'vision' not in name:
                  target_model_name = name
                  break
        
        logger.info(f"Selected model: {target_model_name}")
        model = genai.GenerativeModel(target_model_name)
        
        prompt = """
        Listen to this audio clip and extract the names of all the restaurants, cafes, or eateries mentioned.
        Also, infer the type of cuisine it is based on the context.
        Return ONLY a raw JSON array of objects, where each object has a 'name' (string) and 'cuisine' (string).
        If no restaurants are found, return an empty array [].
        Do not include markdown blocks like ```json. Just raw text.
        """
        
        logger.info("Generating content...")
        response = model.generate_content([prompt, uploaded_file])
        
        # Cleanup file from Google's servers
        try:
             genai.delete_file(uploaded_file.name)
        except Exception as e:
             logger.warning(f"Failed to delete file from Gemini: {e}")

        # Parse the JSON response
        result_text = response.text.strip()
        # Clean up potential markdown formatting if Gemini included it despite instructions
        if result_text.startswith("```json"):
             result_text = result_text[7:-3]
        elif result_text.startswith("```"):
             result_text = result_text[3:-3]
             
        restaurants = json.loads(result_text)
        logger.info(f"Extracted restaurants: {restaurants}")
        return restaurants
        
    except json.JSONDecodeError:
        logger.error(f"Failed to parse Gemini output as JSON: {response.text}")
        return []
    except Exception as e:
        logger.error(f"Gemini API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Extraction failed: {str(e)}")

def geocode_restaurant(name: str) -> dict | None:
    """Uses Google Places API (Text Search) to find coordinates and rating."""
    if not GOOGLE_MAPS_API_KEY:
         logger.warning("No Google Maps API Key provided for geocoding.")
         # Fallback mock coordinates if user didn't test with API key yet
         return {"lat": 37.7749, "lng": -122.4194, "rating": "New"}

    logger.info(f"Geocoding: {name}")
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": name,
        "key": GOOGLE_MAPS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data.get("status") == "OK" and len(data.get("results", [])) > 0:
            place = data["results"][0]
            lat = place["geometry"]["location"]["lat"]
            lng = place["geometry"]["location"]["lng"]
            # Formatting rating carefully
            rating_val = place.get("rating")
            rating = f"{rating_val} ★" if rating_val else "New"
            
            return {
                "lat": lat,
                "lng": lng,
                "rating": rating
            }
        else:
            logger.warning(f"Could not find location for '{name}': {data.get('status')}")
            return None
    except Exception as e:
        logger.error(f"Geocoding failed for {name}: {e}")
        return None

# --- API Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/parse-reel")
def parse_reel(request: ReelRequest):
    output_file = "audio.mp3"
    try:
        # 1. Download Audio
        audio_path = download_audio(request.url, output_file)
        
        # 2. Extract Names with Gemini
        extracted_data = extract_restaurants_with_gemini(audio_path)
        
        if not extracted_data:
             return {"message": "No restaurants found in the audio.", "restaurants": []}
             
        # 3. Geocode with Google Maps
        final_restaurants = []
        for item in extracted_data:
             # Basic structure
             rest = {
                 "id": len(final_restaurants) + 100, # arbitrary id offset
                 "name": item.get("name", "Unknown Restaurant"),
                 "cuisine": item.get("cuisine", "Restaurant"),
             }
             
             # Fetch lat/lng
             location_data = geocode_restaurant(rest["name"])
             if location_data:
                 rest.update(location_data)
                 final_restaurants.append(rest)
             else:
                 # Skip if we can't map it, or you could provide placeholder coords
                 logger.info(f"Skipping {rest['name']} as it couldn't be geocoded.")
                 
        return {"restaurants": final_restaurants}
        
    finally:
        # Cleanup
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass
