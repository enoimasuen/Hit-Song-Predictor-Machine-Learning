# Hit Song Predictor: Amapiano & Afrobeats

Can machine learning predict the next Afrobeats or Amapiano hit?

This project explores whether streaming data, chart performance, social virality, lyric sentiment, and song metadata can help forecast a song's success in two genres shaping the global soundscape.

---

## Overview

This end-to-end machine learning project pulls from the Spotify API and cross-references TikTok virality, Billboard Africa rankings, daily stream rates, and lyric sentiment to classify whether a song is likely to be a "hit." It is built around two genre-specific classification frameworks — one for **Afrobeats** and one for **Amapiano** — each with its own hit-definition thresholds.

---

## Data Sources

- **Spotify API** (`Spotipy`): Playlist-level track metadata and popularity scores across 10 Afrobeats / Amapiano playlists
- **Custom Streaming Dataset**: Total stream counts and release dates (Nigeria & South Africa charts), converted into a streams-per-day (SPD) velocity metric
- **TikTok Viral Flag**: Binary flag (1 = trending) based on video uses relative to days since release
- **Billboard Africa**: Chart-presence confirmation, matched to tracks via fuzzy string matching (`fuzzywuzzy` / `rapidfuzz`)
- **Lyrics** (`lyricsgenius` + VADER): Lyrics pulled from the Genius API and scored for sentiment (−1 to +1) with VADER; cached to `data/lyrics_cache.csv` for reproducibility
- **Genre**: Manually tagged as Afrobeats or Amapiano
- **Song-level metadata**: Tempo (BPM), key, beat strength, mood, and duration

---

## Methodology

**Data Collection & Feature Engineering**
- Extracted track metadata and popularity scores from Spotify using `Spotipy`
- Normalized song titles and merged the Spotify, streaming, TikTok, Billboard, and lyric sources via fuzzy matching
- Pulled lyrics from Genius and derived a `lyric_sentiment` feature with VADER (cached after the first run; missing scores filled with the genre median)
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
- Genre-specific models, evaluated using 5-fold stratified cross-validation
- Re-ran the Random Forests with `lyric_sentiment` added to test whether lyrical tone improves hit detection
- Exploratory data analysis with feature-importance and correlation analysis

---

## Repository

```
.
├── Hit_Song_Predictor_EI.ipynb        # Full notebook: collection, feature engineering, modeling, analysis
├── playlist_and_all_audio_done.csv    # Main dataset: Spotify popularity, SPD, TikTok, Billboard, audio features, genre
├── data/
│   └── lyrics_cache.csv               # Cached Genius lyrics + VADER sentiment scores
└── README.md
```

---

## Results

- **Best model:** Random Forest with class weighting, evaluated via 5-fold stratified cross-validation.
- **Afrobeats hits (minority class):** F1 0.94 (precision 1.00, recall 0.89)
- **Amapiano hits (minority class):** F1 0.91 (precision 1.00, recall 0.83)
- Random Forest outperformed genre-specific Logistic Regression on minority-class (hit) detection in cross-validation (RF hit-class F1 0.94 / 0.91 vs LR ≈ 0.78 / 0.83 for Afrobeats / Amapiano).
- Adding `lyric_sentiment` ranked it among the top features for Afrobeats (above TikTok virality and Billboard presence), but it **did not improve** hit detection — it slightly lowered Afrobeats hit-class F1 and left Amapiano unchanged, likely due to the small sample and VADER's inability to score non-English lyrics.

---

## Limitations

- **Label leakage:** Because the hit label is defined partly from `streams_per_day` and `popularity`, the high importance of those two features is somewhat circular — the model is partly predicting hits from the same signals used to define them. The more meaningful result is that *non-defining* signals (TikTok virality, Billboard presence) outranked musical attributes such as beat strength and tempo. A stronger test of true predictability would forecast hits using only song features and early virality signals, excluding the streaming metrics that overlap with the label.
- **Small sample:** roughly 30 labeled songs per genre, with few hits — these results point to a promising signal, not a production-grade model.
- **VADER language coverage:** VADER is English-trained, so it scores Afrobeats' English/Pidgin lyrics but reads Amapiano's Zulu/Xhosa lyrics unreliably (44 of 60 songs returned real sentiment scores).

---

## Key Takeaways

- Built tailored hit-detection frameworks for Afrobeats and Amapiano, combining signals from the Spotify API, TikTok virality, stream velocity, Billboard Africa rankings, and lyric sentiment.
- Top predictors were **streams-per-day and Spotify popularity**, followed by TikTok virality and Billboard presence.
- **Shareability and visibility signals predicted hits better than the song's sonic features** — audio structure (tempo, duration, key, beat strength) showed weak standalone predictive power.
- Afrobeats hits tended toward **slower tempos**, while Amapiano hits favored steady mid-tempo moods — patterns distinct from U.S. pop hit predictors.
- Lyric sentiment ranked among the top features for Afrobeats but did not improve hit detection, and is limited by VADER's inability to read non-English (Zulu/Xhosa) lyrics.

---

## Future Work

- Improve lyric-sentiment scoring for non-English lyrics, since many Amapiano tracks are in Zulu/Xhosa that VADER scores unreliably.
- Re-frame the task to predict hits from song features and early virality only, removing the streaming metrics used to define the label (a leakage-free test of true predictability).
- Incorporate playlist placement, release timing, and artist reputation as upstream exposure signals.
- Extend to additional African subgenres and explore gradient boosting / model stacking.
- Grow the labeled dataset, since ~30 songs per genre limits how far these conclusions can be pushed.

---

## Contact

Created by **Enoghayin Jillient Imasuen**
Connect via [LinkedIn](https://www.linkedin.com/in/enoghayin-jillient-imasuen-9080b2236) or GitHub [@enoimasuen](https://github.com/enoimasuen)
