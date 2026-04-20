#  Hit Song Predictor: Amapiano & Afrobeats

Can machine learning predict the next Afrobeats or Amapiano hit?  
This project explores whether audio features, streaming data, and metadata can help forecast a song's success in genres that are shaping the global soundscape.

---

##  Overview

This end-to-end machine learning project pulls from the Spotify API, cross-references TikTok virality, Billboard Africa rankings, and daily stream rates to build a predictive model for determining whether a song is likely to be a "hit."

The project is built around two custom classification frameworks — one for **Afrobeats** and one for **Amapiano** — with different thresholds for virality.

---

##  Data Sources

-  **Spotify API**: Playlist-level audio features and popularity scores  
-  **Custom Streaming Dataset**: Streams-per-day stats from Nigeria & South Africa charts  
-  **TikTok Viral Flag**: Binary flag (1 = trending, >30K uses)  
-  **Billboard Africa Top 100**: Hit confirmation  
-  **Genre**: Manually tagged as Afrobeats or Amapiano

---

##  Methodology

- **Data Collection & Feature Engineering**
  - Extracted metadata and audio features from Spotify using `Spotipy`
  - Created binary hit labels using custom business logic (see below)
  - Normalized song titles and merged streaming/tiktok/billboard sources

- **Hit Song Labeling Rules**  
  - **Afrobeats** song = hit if it satisfies 3 of:
    - Spotify Popularity ≥ 73
    - TikTok viral (≥30k uses)
    - Streams/day ≥ 150,000
    - On Billboard Africa charts
  - **Amapiano** song = hit if it satisfies **all 3**:
    - Popularity ≥ 65
    - TikTok viral
    - Streams/day ≥ 60,000

- **Modeling**
  - Baseline Logistic Regression  
  - Random Forest Classifier  
  - Exploratory data analysis (EDA) with feature importance

---

##  Repository
/data/

├── working_spotify_playlist_popularity.csv # Spotify metadata & popularity

├── afrobeats_amapiano_stream_data.csv # Custom stream counts + genre + release date

├── billboard_hits.csv # Billboard Africa chart placements

├── viral_on_tiktok.csv # Binary flag for TikTok virality

/notebooks/

├── hit_song_predictor.ipynb # Full notebook with data collection, modeling, and analysis

📄 README.md 


---

## 🎧 Tracklist (Amapiano & Afrobeats Mix)

This project includes a curated 8-song mix used as inspiration and demonstration:
- *Jealousy* – Khalil Harrison  
- *Shake Ah* – Tyla  
- *Woman* – Rema  
- *Zenzele* – Uncle Waffles  
- *No Competition* – Davido ft. Asake  
- *Bunda* – Musa Keys  
- *Laho* – Shalipopi  
- *My Darling* – Chella  

---

##  Key Takeaways

- Developed tailored hit-detection frameworks for Afrobeats and Amapiano, combining signals from the Spotify API, TikTok virality, stream velocity, and Billboard Africa rankings.
- Enriched the dataset with audio mood indicators by integrating Spotify audio features and lyrical sentiment scraped from Genius.
- Trained both logistic regression and random forest classifiers with genre-aware thresholds and class weighting, achieving **94% F1-score for Afrobeats** and **92% for Amapiano**.
- Found that Afrobeats hits tend to have **slower tempos**, while Amapiano hits favor **mellow, steady moods** — revealing patterns distinct from U.S. pop hit predictors.
- Discovered that TikTok virality was a powerful predictor of Billboard presence, especially when combined with streaming velocity and Spotify popularity metrics.
- Identified danceability, energy, and acousticness as genre-specific drivers of hit probability, with notable divergence between Afrobeats and Amapiano sonic signatures.


## 📬 Contact

Created by **Enoghayin Jillient Imasuen**  
Feel free to connect via [LinkedIn](https://www.linkedin.com/in/enoghayin-jillient-imasuen-9080b2236?) or GitHub at [enoimasuen](https://github.com/enoimasuen)

