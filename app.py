import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 🎵 Spotify Setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="79313691919c476fb9cf995fcae929a4",
    client_secret="f082bbd362c141798be1f157b7fc28f2"
))

# 🤗 Hugging Face Setup
hf_token = "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"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", use_auth_token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", use_auth_token=hf_token)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 🎧 Mood emojis
mood_emojis = {
    "Happy": "😄",
    "Sad": "😢",
    "Energetic": "⚡",
    "Relaxed": "🌙"
}

# Load CSV
df = pd.read_csv("mood_music.csv")

# 🎨 Page config
st.set_page_config(page_title="🎧 Mood Music AI", layout="wide", page_icon="🎵")

# 🎨 CSS Styling
st.markdown('''
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1505678261036-a3fcc5e884ee");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
h1 {
    font-size: 3em;
    color: white;
    text-align: center;
}
div.stButton > button {
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    border-radius: 10px;
}
</style>
''', unsafe_allow_html=True)

# 🎤 Detect mood from text using Hugging Face
def detect_mood_from_text(text):
    try:
        result = classifier(text)[0]
        label = result["label"].lower()
        if "positive" in label:
            return "Happy"
        elif "negative" in label:
            return "Sad"
        else:
            return "Relaxed"
    except Exception as e:
        st.warning(f"🤖 AI mood detection failed: {e}")
        return None

# 🎧 Spotify search
def get_spotify_recommendations(mood, limit=5):
    query = f"{mood} mood"
    results = sp.search(q=query, type='track', limit=limit)
    return results['tracks']['items']

# 🧠 App Header
st.markdown("<h1>🎧 Mood-Based Music Recommender</h1>", unsafe_allow_html=True)
st.markdown("## 🧠 Describe your mood or choose manually to get your perfect playlist.")

# Mood input
st.markdown("#### 📝 Describe your current mood in words:")
user_input = st.text_input("How are you feeling today? (e.g., I'm feeling pumped up for the gym!)")
detected_mood = detect_mood_from_text(user_input) if user_input else None

# Choose mood
mood = detected_mood or st.selectbox("🎭 Or choose your mood manually:", df["mood"].unique())

# Store mood history
if "mood_history" not in st.session_state:
    st.session_state["mood_history"] = []

# 🎶 Music Recommendations
if mood:
    st.session_state["mood_history"].append(mood.capitalize())
    emoji = mood_emojis.get(mood.capitalize(), "🎵")
    st.success(f"{emoji} Great! Here are your recommendations.")

    # Local CSV music
    filtered = df[df["mood"].str.lower() == mood.lower()]
    if not filtered.empty:
        st.subheader("🎵 Songs from local list:")
        for _, row in filtered.iterrows():
            st.markdown(f"**{row['song']}** by *{row['artist']}*")
            if isinstance(row['youtube'], str) and "v=" in row['youtube']:
                video_id = row['youtube'].split("v=")[-1]
                st.video(f"https://www.youtube.com/embed/{video_id}")

    # Spotify recommendations
    st.subheader("🎧 Spotify Recommendations:")
    try:
        spotify_tracks = get_spotify_recommendations(mood)
        for track in spotify_tracks:
            st.markdown(f"[{track['name']} - {track['artists'][0]['name']}]({track['external_urls']['spotify']})")
            st.image(track['album']['images'][0]['url'], width=200)
    except:  # noqa: E722
        st.warning("⚠️ Could not fetch Spotify tracks.")

    # Custom song input
    st.subheader("➕ Add Your Favorite Song (Session Only)")
    with st.form("user_song_form"):
        user_song = st.text_input("Song Name")
        user_artist = st.text_input("Artist Name")
        user_link = st.text_input("YouTube Link")
        submit = st.form_submit_button("Add Song")
        if submit and user_song and user_artist and user_link:
            st.markdown(f"**{user_song}** by *{user_artist}*")
            try:
                user_video_id = user_link.split("v=")[-1]
                st.video(f"https://www.youtube.com/embed/{user_video_id}")
            except:  # noqa: E722
                st.error("⚠️ Invalid YouTube link.")

    st.markdown("---")
    st.info("🕓 Moods this session: " + ", ".join(st.session_state["mood_history"]))

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: white;'>🎶 Built with ❤️ by <b>Prashanthi Mallepula</b> using Streamlit & Hugging Face</p>", unsafe_allow_html=True)
