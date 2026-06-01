# Hit Song Predictor: Amapiano & Afrobeats

Can machine learning predict the next Afrobeats or Amapiano hit?

This project explores whether streaming data, chart performance, social virality, and song metadata can help forecast a song's success in two genres shaping the global soundscape.

---

## Overview

This end-to-end machine learning project pulls from the Spotify API and cross-references TikTok virality, Billboard Africa rankings, and daily stream rates to classify whether a song is likely to be a "hit." It is built around two genre-specific classification frameworks — one for **Afrobeats** and one for **Amapiano** — each with its own hit-definition thresholds.

---

## Data Sources

- **Spotify API** (`Spotipy`): Playlist-level track metadata and popularity scores across 10 Afrobeats / Amapiano playlists
- **Custom Streaming Dataset**: Total stream counts and release dates (Nigeria & South Africa charts), converted into a streams-per-day (SPD) velocity metric
- **TikTok Viral Flag**: Binary flag (1 = trending) based on video uses relative to days since release
- **Billboard Africa**: Chart-presence confirmation, matched to tracks via fuzzy string matching (`fuzzywuzzy`)
- **Genre**: Manually tagged as Afrobeats or Amapiano
- **Song-level metadata**: Tempo (BPM), key, beat strength, mood, and duration

---

## Methodology

**Data Collection & Feature Engineering**
- Extracted track metadata and popularity scores from Spotify using `Spotipy`
- Normalized song titles and merged the Spotify, streaming, TikTok, and Billboard sources
- Created binary hit labels using the custom business logic below

**Hit Song Labeling Rules**
- **Afrobeats** = hit if it satisfies **3 of 4**:
  - Spotify popularity ≥ 73
  - TikTok viral
  - Streams/day ≥ 300,000
  - On Billboard Africa
- **Amapiano** = hit if it satisfies **all 3**:
  - Spotify popularity ≥ 65
  - TikTok viral
  - Streams/day ≥ 75,000

**Modeling**
- Baseline Logistic Regression (with stratified splits)
- Random Forest Classifier with `class_weight='balanced'` to handle hit / non-hit imbalance
- Evaluated using 5-fold stratified cross-validation
- Exploratory data analysis with feature-importance and correlation analysis

---

## Repository

```
/data/
├── working_spotify_playlist_popularity.csv   # Spotify metadata & popularity
├── afrobeats_amapiano_stream_data.csv        # Stream counts + genre + release date
├── billboard_hits.csv                        # Billboard Africa chart placements
├── viral_on_tiktok.csv                       # Binary flag for TikTok virality
/notebooks/
├── hit_song_predictor.ipynb                  # Full notebook: collection, modeling, analysis
README.md
```

---

## Results

- **Best model:** Random Forest with class weighting, evaluated via 5-fold stratified cross-validation.
- **Afrobeats hits:** F1 0.94 (precision 1.00, recall 0.89)
- **Amapiano hits:** F1 0.92 (precision 0.86, recall 1.00)
- Random Forest outperformed Logistic Regression on minority-class (hit) detection across both genres.

---

## Key Takeaways

- Built tailored hit-detection frameworks for Afrobeats and Amapiano, combining signals from the Spotify API, TikTok virality, stream velocity, and Billboard Africa rankings.
- Top predictors were **streams-per-day and Spotify popularity**, followed by TikTok virality and Billboard presence.
- Audio structure (tempo, duration) showed weak standalone predictive power — shareability and visibility signals predicted hits better than the song's sonic features.
- Afrobeats hits tended toward **slower tempos**, while Amapiano hits favored steady mid-tempo moods — patterns distinct from U.S. pop hit predictors.

---

## Future Work

- Add full NLP sentiment analysis of lyrics (e.g., scraped from Genius) to enrich the current mood tagging.
- Incorporate playlist placement, release timing, and artist reputation as upstream exposure signals.
- Extend to additional African subgenres and explore gradient boosting / model stacking.

---

## Contact

Created by **Enoghayin Jillient Imasuen**
Connect via [LinkedIn](https://www.linkedin.com/in/enoghayin-jillient-imasuen-9080b2236) or GitHub [@enoimasuen](https://github.com/enoimasuen)
