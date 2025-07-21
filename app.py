import streamlit as st
import requests
import joblib
import pandas as pd

st.set_page_config(page_title="⏱️ Tillægstid Predictor", layout="wide")

API_TOKEN = "din_sportmonks_token"
BASE_URL = "https://api.sportmonks.com/v3/football"
MODEL = joblib.load("added_time_model.joblib")

def get_live_fixtures():
    url = f"{BASE_URL}/fixtures/live"
    resp = requests.get(url, params={"api_token": API_TOKEN, "include": "teams;scores;events;commentaries"})
    return resp.json().get("data", []) if resp.status_code == 200 else []

def score_comment(text):
    w = {
        "injury": 3.0, "stretcher": 3.5, "var": 1.5,
        "delay": 1.5, "red card": 2.0, "medical": 2.5,
    }
    text = text.lower()
    return sum(v for k, v in w.items() if k in text)

def extract_features(fixture, half):
    events = fixture.get("events", {}).get("data", [])
    comments = fixture.get("commentaries", {}).get("data", [])
    def in_half(m): return (half == 1 and m <= 45) or (half == 2 and m > 45)
    return {
        "goals": sum(e["type"]=="goal" and in_half(e["minute"]["minute"]) for e in events),
        "substitutions": sum(e["type"]=="substitution" and in_half(e["minute"]["minute"]) for e in events),
        "yellow_cards": sum(e["type"]=="yellowcard" and in_half(e["minute"]["minute"]) for e in events),
        "red_cards": sum(e["type"]=="redcard" and in_half(e["minute"]["minute"]) for e in events),
        "var_events": sum(e["type"]=="var" and in_half(e["minute"]["minute"]) for e in events),
        "injury_events": sum(e["type"]=="injury" and in_half(e["minute"]["minute"]) for e in events),
        "nlp_score": sum(score_comment(c["comment"]) for c in comments if in_half(c["minute"]["minute"]))
    }

if "watched" not in st.session_state:
    st.session_state.watched = set()
if "notified" not in st.session_state:
    st.session_state.notified = set()

st.title("⭐ Tillægstid Predictor – Live Kampe")
live = get_live_fixtures()

for fixture in live:
    fid = str(fixture["id"])
    home = fixture["teams"]["data"][0]
    away = fixture["teams"]["data"][1]
    minute = fixture.get("time", {}).get("minute", 0)
    score = fixture.get("scores", {}).get("data", {})
    hs, as_ = score.get("home_score", "?"), score.get("away_score", "?")

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.image(home["image_path"], width=50)
        st.write(home["name"])
    with col2:
        st.write(f"**{minute}′**  —  **{hs}–{as_}**")
        if fid in st.session_state.watched:
            if st.button("⭐ Fjern", key="rm"+fid):
                st.session_state.watched.remove(fid)
        else:
            if st.button("☆ Overvåg", key="add"+fid):
                st.session_state.watched.add(fid)
    with col3:
        st.image(away["image_path"], width=50)
        st.write(away["name"])

    st.markdown("---")

for fixture in live:
    fid = str(fixture["id"])
    minute = fixture.get("time", {}).get("minute", 0)
    if fid not in st.session_state.watched:
        continue
    for half, thresh in [(1,40),(2,85)]:
        key = fid+f"_{half}"
        if minute>=thresh and key not in st.session_state.notified:
            feats = extract_features(fixture, half)
            X = [feats[k] for k in MODEL.feature_names_in_]
            pred = round(MODEL.predict([X])[0])
            st.success(f"🔔 Tillægstid i halvleg {half}: {pred} min")
            st.session_state.notified.add(key)
