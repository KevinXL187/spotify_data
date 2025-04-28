import pandas as pd
from pathlib import Path
import spotipy, fasttext
from spotipy.oauth2 import SpotifyClientCredentials

TOP_N = 1000 
INPUT_CSV = "spotify_data.csv" 
OUTPUT_CSV = f"spotify_top{TOP_N}_with_language.csv"
SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""

def load_lang_model():
    model_path = "lid.176.bin"
    if not Path(model_path).exists():
        raise FileNotFoundError(
            "Download FastText model first: "
            "https://fasttext.cc/docs/en/language-identification.html"
        )
    return fasttext.load_model(model_path)

def clean_data(df):pass

def aggregate_plays(df):
    # Convert to numeric and clean
    df['ms_played'] = pd.to_numeric(df['ms_played'], errors='coerce')
    df = df.dropna(subset=['ms_played'])
    
    # Aggregate by track
    agg_df = df.groupby([
        'master_metadata_album_artist_name',
        'master_metadata_track_name'
    ]).agg({
        'ms_played': 'sum'
    }).reset_index()
    
    return agg_df

def filter_topN(df):
    agg_df = aggregate_plays(df)
    return agg_df.nlargest(TOP_N, 'ms_played')

def predict_language(text, model): 
    try:
        print(text)
        if not isinstance(text, str) or len(text.strip()) < 2:
            return "Unknown"
        
        # Clean text (remove special chars, keep CJK + letters)
        clean_text = ''.join(c for c in text if c.isalpha() or c.isspace() or ord(c) > 127)
        
        # Predict with FastText
        (lang,), (conf,) = model.predict(clean_text, k=1)
        lang = lang.replace("__label__", "")
        
        if conf >= 0.7 : return lang.capitalize() 
        else: return 'EN'
            
    except Exception as e:
        print(f"Language detection error for '{text}': {str(e)}")
        return "Unknown"

def get_genre(artist_name, sp):
    # Get only the first genre of the first result
    try:
        results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
        if results["artists"]["items"]:
            return results["artists"]["items"][0].get("genres", ["Unknown"])[0]
    except: return "Unknown"

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
    top_df = filter_topN(df)

    model = load_lang_model()
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))

    # add genre
    print("Fetching genres...")
    top_df["genre"] = top_df["master_metadata_album_artist_name"].apply(
        lambda x: get_genre(x, sp) if sp else "Unknown"
    )

    # Add language column
    print("Detecting languages...")
    top_df["language"] = top_df.apply(
        lambda row: predict_language(
            f"{row['master_metadata_album_artist_name']} {row['master_metadata_track_name']}",
            model
        ),
        axis=1
    )

    top_df.to_csv(OUTPUT_CSV, index=False)
    # Print summary
    print("\nLanguage distribution:")
    print(top_df["language"].value_counts())
    print("\nTop genres:")
    print(top_df["genre"].value_counts().head(10))