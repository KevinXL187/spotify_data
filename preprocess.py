import pandas as pd
import pathlib as Path
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import fasttext, requests


TOP_N = 1000 
INPUT_CSV = "spotify_data.csv" 
OUTPUT_CSV = f"spotify_top{TOP_N}_with_language.csv"
SPOTIFY_CLIENT_ID =""
SPOTIFY_CLIENT_SECRET =""

def load_model():
    model_path = "lid.176.bin"
    if not Path(model_path).exists():
        raise FileNotFoundError(
            "Download FastText model first: "
            "https://fasttext.cc/docs/en/language-identification.html"
        )
    return fasttext.load_model(model_path)

def filter_topN(df):
    if "msPlayed" in df.columns:
        return df.nlargest(TOP_N, "msPlayed")
    else:
        raise KeyError("CSV needs 'msPlayed' or 'count' column to filter top tracks.")

def predict_language(text, model): 
    try:
        lang = model.predict(text)[0][0].replace("__label__", "")
        return {"zh": "Chinese", "ja": "Japanese", "ko": "Korean"}.get(lang, "Other")
    except:
        return "Unknown"

def get_genre(artist_name, sp):
    try:
        results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
        if results["artists"]["items"]:
            return results["artists"]["items"][0].get("genres", ["Unknown"])[0]
    except: return "Unknown"

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV)
    top_df = filter_topN(df)

    model = load_model()
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))

    # Add language column
    top_df["language"] = (top_df["artist_name"] + " " + top_df["track_name"]).apply(
        lambda x: predict_language(x, model)
    )

    # add genre
    top_df["genre"] = top_df["artist_name"].apply(
            lambda x: get_genre(x, sp)
        )


    top_df.to_csv(OUTPUT_CSV, index=False)
    