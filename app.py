import streamlit as st
from music_recommender3 import MusicRecommender

# Add page configuration
st.set_page_config(
    page_title="Music Recommender App",
    page_icon="🎵",
    layout="wide"
)

# Add CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .error-font {
        color: #ff0000;
    }
    .success-font {
        color: #00ff00;
    }
    </style>
    """, unsafe_allow_html=True)

# Spotify and Discogs API credentials
SPOTIFY_CLIENT_ID = "26c65df3e5844f1dbe355d82d80c9f6f"
SPOTIFY_CLIENT_SECRET = "2d4d2b147bc942b999564a5e8649b987"
DISCOGS_TOKEN = "FbbkQDyGoGsJlnSqVfFwqfvUWnrtDcBiWmyHOHjX"

# Main app
st.title("🎵 Music Recommender App")
st.markdown("Enter a Discogs release URL to get music recommendations")

# Input for Discogs release URL
discogs_url = st.text_input(
    "Discogs Release URL:",
    placeholder="https://www.discogs.com/release/..."
)

def initialize_recommender():
    """Initialize the recommender only when needed."""
    with st.spinner('Initializing recommender system...'):
        try:
            recommender = MusicRecommender(
                spotify_client_id=SPOTIFY_CLIENT_ID,
                spotify_client_secret=SPOTIFY_CLIENT_SECRET,
                discogs_token=DISCOGS_TOKEN
            )
            return recommender
        except Exception as e:
            st.error(f"Failed to initialize recommender: {str(e)}")
            return None

# Only process URL if one is provided
if discogs_url:
    # Initialize recommender when URL is provided
    if 'recommender' not in st.session_state:
        st.session_state.recommender = initialize_recommender()
    
    if st.session_state.recommender:
        try:
            with st.spinner('Analyzing album...'):
                # Validate URL format
                if not discogs_url.startswith('https://www.discogs.com/'):
                    st.error("Please enter a valid Discogs URL")
                else:
                    # Get album analysis
                    analysis = st.session_state.recommender.analyze_album(discogs_url)
                    
                    # Create two columns for layout
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Album Information
                        st.header("Album Details")
                        discogs_info = analysis['discogs_info']
                        
                        # Display album image if available
                        spotify_info = analysis['spotify_info']
                        if spotify_info.get('album_image'):
                            st.image(spotify_info['album_image'], width=300)
                        
                        # Album metadata
                        st.markdown(f"**Artist**: {discogs_info.get('artist', 'Unknown Artist')}")
                        st.markdown(f"**Album**: {discogs_info.get('album', 'Unknown Album')}")
                        st.markdown(f"**Label**: {discogs_info.get('label', 'Unknown Label')}")
                        st.markdown(f"**Catalog**: {discogs_info.get('catalog', 'Unknown')}")
                        st.markdown(f"**Format**: {discogs_info.get('format', 'Unknown Format')}")
                        st.markdown(f"**Year**: {discogs_info.get('year', 'Unknown Year')}")
                        st.markdown(f"**Styles**: {', '.join(discogs_info.get('styles', []))}")
                        
                        # Market Prices
                        st.subheader("Market Prices")
                        prices = {k: v for k, v in discogs_info.items() if k in ['low', 'median', 'high']}
                        if any(v is not None and v != 'N/A' for v in prices.values()):
                            for key, value in prices.items():
                                if value and value != 'N/A':
                                    st.markdown(f"**{key.capitalize()}**: {value}")
                        else:
                            st.markdown("*No price information available*")
                    
                    with col2:
                        # Audio Analysis
                        st.header("Audio Analysis")
                        analysis_info = analysis['analysis']
                        st.markdown(f"**Average BPM**: {analysis_info.get('mean_bpm', 'N/A'):.1f}")
                        st.markdown(f"**Predominant Key**: {analysis_info.get('key', 'N/A')}")
                        st.markdown(f"**Cluster**: {analysis_info.get('cluster', 'N/A')}")
                        
                        # Audio Features
                        st.subheader("Audio Features")
                        for feature, value in analysis_info.get('audio_features', {}).items():
                            st.markdown(f"**{feature.capitalize()}**: {value:.3f}")
                    
                    # Visualizations
                    st.header("Audio Features Profile")
                    fig = st.session_state.recommender.visualize_album_features(spotify_info['tracks'])
                    st.pyplot(fig)
                    
                    # Recommendations
                    st.header("Recommended Tracks")
                    recommendations = analysis['recommendations']
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {rec.get('track_title', 'Unknown Title')} by {rec.get('artist_name', 'Unknown Artist')}"):
                            st.markdown(f"**Reason**: {rec.get('recommendation_reason', 'No reason provided')}")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Style Match**: {rec.get('style_similarity', 'N/A'):.2f}")
                            with col2:
                                st.markdown(f"**Audio Similarity**: {rec.get('audio_similarity', 'N/A'):.2f}")
                            with col3:
                                st.markdown(f"**Overall Score**: {rec.get('final_score', 'N/A'):.2f}")
                            
                            if rec.get('preview_url'):
                                st.audio(rec['preview_url'], format='audio/mp3')

        except Exception as e:
            st.error(f"Error analyzing album: {str(e)}")
            st.markdown("Please check the URL and try again.")
else:
    st.info("👆 Enter a Discogs URL above to start the analysis")