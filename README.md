# Music Recommender App 🎵

## Overview
The Music Recommender App is a personalized music recommendation system developed using Python and Streamlit. This app leverages the Spotify and Discogs APIs to provide users with music recommendations based on the analysis of album data. By inputting a Discogs release URL, users receive a detailed analysis of the album, visualizations of audio features, and personalized track recommendations.

## ✨ Features
- **Discogs and Spotify Integration**: Retrieves album information from Discogs and audio features from Spotify
- **Audio Analysis**: Provides analysis of audio features like danceability, energy, tempo, and more
- **Album Visualization**: Generates visualizations for album audio features and track tempos
- **Music Recommendations**: Offers personalized music recommendations based on similarity in audio features, genre, and style
- **Market Price Information**: Displays pricing statistics for albums listed on Discogs (when available)

## 🗂 Project Structure
- `app.py`: Main Streamlit app file - manages interface and integrates with MusicRecommender
- `music_recommender3.py`: Core recommendation logic, audio feature analysis, and API integrations
- `price_scrapping.ipynb`: Jupyter notebook for Discogs price scraping
- Data directories (not in repo):
  - `data/`
  - `data-extracts/`
  - `complete/`
  - `split_by_genre/`
  - `split_by_genre_no_subgenre/`

## 🛠 Requirements
### Technologies
- Python 3.8+

### Libraries
- Streamlit
- Spotipy
- Discogs_client
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Requests
- BeautifulSoup
- Selenium

### APIs
- Spotify API
- Discogs API

## 🚀 Usage
1. Launch the app
2. Enter a Discogs release URL
3. View detailed album information:
   - Artist details
   - Genre and style
   - Release year
   - Market prices
4. Explore audio analysis visualizations
5. Get five personalized music recommendations with preview links

## 🔮 Future Improvements
- [ ] Add genre-specific recommendations
- [ ] Use related artists to expand recommendations
- [ ] Enhance error handling
- [ ] Optimize performance

## Links
- Presentation : https://docs.google.com/presentation/d/1IXFF2OvVF8Krz0sYycRnMDoCh-OUlgMf6Ges8M1iiBE/edit#slide=id.g312729603b6_2_12
- Kaggle : https://www.kaggle.com/datasets/mrmorj/spotify-audio-features-2023