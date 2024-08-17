import re
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np



# Load CSV data
spotify_df = pd.read_csv('spotify_songs.csv')

# Normalize continuous features
continuous_features = ['danceability', 'energy', 'speechiness', 
                       'acousticness', 'instrumentalness', 'valence', 'track_popularity']

categorical_features = ['track_artist', 'playlist_genre', 'playlist_subgenre']

scaler = MinMaxScaler()
spotify_df[continuous_features] = scaler.fit_transform(spotify_df[continuous_features])

# Preserve original names for checking existence before encoding
original_artists = spotify_df['track_artist'].str.lower().str.strip()
original_genres = spotify_df['playlist_genre'].str.lower().str.strip()
original_subgenres = spotify_df['playlist_subgenre'].str.lower().str.strip()

# Ensure 'playlist_genre' and 'playlist_subgenre' are strings and lowercased before encoding

# Encode categorical features with separate encoders
encoder_artist = LabelEncoder()
encoder_genre = LabelEncoder()
encoder_subgenre = LabelEncoder()

spotify_df['track_artist'] = encoder_artist.fit_transform(spotify_df['track_artist'])
reverse_artist_mapping = dict(zip(spotify_df['track_artist'], original_artists))

spotify_df['playlist_genre'] = encoder_genre.fit_transform(spotify_df['playlist_genre'])
reverse_genre_mapping = dict(zip(spotify_df['playlist_genre'], original_genres))

spotify_df['playlist_subgenre'] = encoder_subgenre.fit_transform(spotify_df['playlist_subgenre'])
reverse_subgenre_mapping = dict(zip(spotify_df['playlist_subgenre'], original_subgenres))

# Function to encode artist name
def encode_artist(artist_name):
    artist_name = artist_name.lower().strip()
    for encoded, original in reverse_artist_mapping.items():
        if original == artist_name:
            return encoded
    raise ValueError(f"Artist '{artist_name}' not found in the dataset.")

# Function to encode genre name
def encode_genre(genre_name):
    genre_name = genre_name.lower().strip()
    for encoded, original in reverse_genre_mapping.items():
        if original == genre_name:
            return encoded
    raise ValueError(f"Genre '{genre_name}' not found in the dataset.")

# Function to encode subgenre name
def encode_subgenre(subgenre_name):
    subgenre_name = subgenre_name.lower().strip()
    for encoded, original in reverse_subgenre_mapping.items():
        if original == subgenre_name:
            return encoded
    raise ValueError(f"Subgenre '{subgenre_name}' not found in the dataset.")


# Function to collect user preferences into the correct structure
def get_user_preferences(
    artist=None, artist_weight=None,
    popularity=None, popularity_weight=None,
    genre=None, genre_weight=None,
    subgenre=None, subgenre_weight=None,
    danceability=None, danceability_weight=None,
    energy=None, energy_weight=None,
    speechiness=None, speechiness_weight=None,
    acousticness=None, acousticness_weight=None,
    instrumentalness=None, instrumentalness_weight=None,
    valence=None, valence_weight=None
):
    user_preferences = {}

    if artist is not None and artist_weight is not None:
        encoded_artist = encode_artist(artist)
        user_preferences['track_artist'] = {encoded_artist: artist_weight}

    if popularity is not None and popularity_weight is not None:
        user_preferences['track_popularity'] = {popularity: popularity_weight}

    if genre is not None and genre_weight is not None:
        encoded_genre = encode_genre(genre)
        user_preferences['playlist_genre'] = {encoded_genre: genre_weight}

    if subgenre is not None and subgenre_weight is not None:
        encoded_subgenre = encode_subgenre(subgenre)
        user_preferences['playlist_subgenre'] = {encoded_subgenre: subgenre_weight}

    if danceability is not None and danceability_weight is not None:
        user_preferences['danceability'] = {danceability: danceability_weight}

    if energy is not None and energy_weight is not None:
        user_preferences['energy'] = {energy: energy_weight}

    if speechiness is not None and speechiness_weight is not None:
        user_preferences['speechiness'] = {speechiness: speechiness_weight}

    if acousticness is not None and acousticness_weight is not None:
        user_preferences['acousticness'] = {acousticness: acousticness_weight}

    if instrumentalness is not None and instrumentalness_weight is not None:
        user_preferences['instrumentalness'] = {instrumentalness: instrumentalness_weight}

    if valence is not None and valence_weight is not None:
        user_preferences['valence'] = {valence: valence_weight}

    return user_preferences

def get_user_preferences_percentile(
    artist=None, artist_weight=None,
    popularity=None, popularity_weight=None,
    genre=None, genre_weight=None,
    subgenre=None, subgenre_weight=None,
    danceability=None, danceability_weight=None,
    energy=None, energy_weight=None,
    speechiness=None, speechiness_weight=None,
    acousticness=None, acousticness_weight=None,
    instrumentalness=None, instrumentalness_weight=None,
    valence=None, valence_weight=None
):
    user_preferences = {}

    # Map artist to encoded value as usual
    if artist is not None and artist_weight is not None:
        encoded_artist = encode_artist(artist)
        user_preferences['track_artist'] = {encoded_artist: artist_weight}

    # Map percentile inputs to actual values in the dataset
    def map_percentile_to_value(feature, percentile):
        return np.percentile(spotify_df[feature], percentile * 100)

    if popularity is not None and popularity_weight is not None:
        value = map_percentile_to_value('track_popularity', popularity)
        user_preferences['track_popularity'] = {value: popularity_weight}

    if genre is not None and genre_weight is not None:
        encoded_genre = encode_genre(genre)
        user_preferences['playlist_genre'] = {encoded_genre: genre_weight}

    if subgenre is not None and subgenre_weight is not None:
        encoded_subgenre = encode_subgenre(subgenre)
        user_preferences['playlist_subgenre'] = {encoded_subgenre: subgenre_weight}

    if danceability is not None and danceability_weight is not None:
        value = map_percentile_to_value('danceability', danceability)
        user_preferences['danceability'] = {value: danceability_weight}

    if energy is not None and energy_weight is not None:
        value = map_percentile_to_value('energy', energy)
        user_preferences['energy'] = {value: energy_weight}

    if speechiness is not None and speechiness_weight is not None:
        value = map_percentile_to_value('speechiness', speechiness)
        user_preferences['speechiness'] = {value: speechiness_weight}

    if acousticness is not None and acousticness_weight is not None:
        value = map_percentile_to_value('acousticness', acousticness)
        user_preferences['acousticness'] = {value: acousticness_weight}

    if instrumentalness is not None and instrumentalness_weight is not None:
        value = map_percentile_to_value('instrumentalness', instrumentalness)
        user_preferences['instrumentalness'] = {value: instrumentalness_weight}

    if valence is not None and valence_weight is not None:
        value = map_percentile_to_value('valence', valence)
        user_preferences['valence'] = {value: valence_weight}

    return user_preferences

# Function to calculate the similarity score
def calculate_similarity_score(song, user_preferences):
    score = 0
    total_weight = 0

    for characteristic, preference in user_preferences.items():
        for user_value, weight in preference.items():
            total_weight += weight
            if characteristic in continuous_features:
                # Calculate similarity for continuous features (e.g., energy, speechiness)
                score += weight * (1 - abs(song[characteristic] - user_value))
            elif characteristic in categorical_features:
                # Calculate similarity for categorical features (e.g., artist, genre)
                score += weight * (1 if song[characteristic] == user_value else 0)
    
    # Normalize the score by the total weight
    if total_weight > 0:
        score /= total_weight
    return score


def normalize_title(title):
    """
    Normalize song titles by converting to lowercase, removing text inside parentheses, and stripping whitespace.
    Handle cases where title is not a string (e.g., NaN).
    """
    if not isinstance(title, str):
        return ""  # Return an empty string or a placeholder if the title is not a string
    title = title.lower()
    # Remove text inside parentheses
    title = re.sub(r'\(.*?\)', '', title)
    # Remove leading and trailing whitespace
    title = title.strip()
    return title


def get_unique_recommendations(recommended_songs):
    # Normalize the titles
    recommended_songs['normalized_title'] = recommended_songs['track_name'].apply(normalize_title)
    
    # Sort by similarity score to keep the highest one in case of duplicates
    recommended_songs = recommended_songs.sort_values(by='similarity_score', ascending=False)
    
    # Drop duplicates based on normalized title, keeping the first (highest similarity score)
    unique_songs = recommended_songs.drop_duplicates(subset=['normalized_title'])
    
    # Drop the temporary normalized title column
    unique_songs = unique_songs.drop(columns=['normalized_title'])
    
    return unique_songs

# Set the page configuration
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Concise Header
st.title("🎵 Spotify Song Recommender")
st.subheader("""Welcome to the Spotify Song Recommender!""")
st.subheader("""Select your song preferences from the sidebar and adjust their values and weights below to receive personalized song recommendations. Higher weighted preferences are prioritized more in song calculations.""")
st.write('')
st.write("For more precise results, read the advanced tooltips on the GitHub page (found at the top-right of the screen), or try using percentile mode.")
# Checkbox to display histograms of all input variables
if st.checkbox("***Show input variable distributions***"):
    st.write("Distributions of Input Variables:")
    
    # Loop through the continuous features and plot histograms
    for feature in continuous_features:
        fig, ax = plt.subplots()
        ax.hist(spotify_df[feature], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
        
    csv = spotify_df.to_csv(index=False)
    st.download_button(
        label="Download dataset as CSV",
        data=csv,
        file_name='spotify_songs.csv',
        mime='text/csv',
    )

# Initialize variables to None
artist = None
artist_weight = None
popularity = None
popularity_weight = None
genre = None
genre_weight = None
subgenre = None
subgenre_weight = None
danceability = None
danceability_weight = None
energy = None
energy_weight = None
speechiness = None
speechiness_weight = None
acousticness = None
acousticness_weight = None
instrumentalness = None
instrumentalness_weight = None
valence = None
valence_weight = None

# Checkbox to select features
percentile_mode = st.checkbox("***Percentile Mode***", help='Interprets your input numbers as percentiles instead of their face values. This feature helps reduce the skew of the more volatile song traits such as speechiness, acousticness, and instrumentalness.')
st.write('')

with st.sidebar:
    st.write("***Preferences***", )
    track_artist_checkbox = st.checkbox("Artist")
    track_popularity_checkbox = st.checkbox("Popularity")
    playlist_genre_checkbox = st.checkbox("Genre")
    playlist_subgenre_checkbox = st.checkbox("Subgenre")
    danceability_checkbox = st.checkbox("Danceability")
    energy_checkbox = st.checkbox("Energy")
    speechiness_checkbox = st.checkbox("Speechiness")
    acousticness_checkbox = st.checkbox("Acousticness")
    instrumentalness_checkbox = st.checkbox("Instrumentalness")
    valence_checkbox = st.checkbox("Valence")

# Interface widgets to get and store user values
if track_artist_checkbox:
    artist = st.text_input("Artist name", key='artist', help='The name of the artist for the song.')
    artist_cleaned = artist.lower().strip()

    # Check if the artist exists in the original artist list (before encoding)
    if (artist_cleaned != '') and (artist_cleaned not in original_artists.values):
        st.write('Artist not in dataset :(')
    
    artist_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='artist_weight')/100
    st.subheader(' ')
    
if track_popularity_checkbox:
    popularity = st.number_input('Track Popularity (0-100)', min_value=0, max_value=100, key='popularity', help="The song's popularity score (0-100), where higher is better.")/100
    popularity_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='popularity_weight')/100
    st.subheader(' ')

if playlist_genre_checkbox:
    genre = st.selectbox('Genre', ('pop','rap','r&b','rock','latin','edm'), key='genre', help='The genre of the song.')
    genre_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='genre_weight')/100
    st.subheader(' ')
    
if playlist_subgenre_checkbox:
    subgenre = st.selectbox('Subgenre', (
        'dance pop', 'post-teen pop', 'electropop', 'indie poptimism', 
        'hip hop', 'southern hip hop', 'gangster rap', 'trap', 
        'album rock', 'classic rock', 'permanent wave', 'hard rock', 
        'tropical', 'latin pop', 'reggaeton', 'latin hip hop', 
        'urban contemporary', 'hip pop', 'new jack swing', 
        'neo soul', 'electro house', 'big room', 'pop edm', 
        'progressive electro house'), key='subgenre', help="The subgenre of the song.")
    subgenre_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100,key='subgenre_weight')/100
    st.subheader(' ')

# Danceability Checkbox
if danceability_checkbox:
    danceability = st.number_input('Danceability', min_value=0, max_value=100, key='danceability', help="How suitable a track is for dancing, ranging from 0.0 (least danceable) to 1.0 (most danceable).")/100
    danceability_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='danceability_weight')/100
    st.subheader(' ')

if energy_checkbox:
    energy = st.number_input('Energy', min_value=0, max_value=100, key='energy', help="A measure of intensity and activity, from 0.0 to 1.0. Higher energy indicates faster, louder, and noisier tracks.")/100
    energy_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='energy_weight') / 100
    st.subheader(' ')
                            
# Speechiness Checkbox
if speechiness_checkbox:
    speechiness = st.number_input('Speechiness', min_value=0, max_value=100, key='speechiness', help="Detects the presence of spoken words in a track. Values closer to 1.0 indicate more speech-like content.")/100
    speechiness_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='speechiness_weight') / 100
    st.subheader(' ')

# Acousticness Checkbox
if acousticness_checkbox:
    acousticness = st.number_input('Acousticness', min_value=0, max_value=100, key='acousticness', help="A confidence measure of whether the track is acoustic, ranging from 0.0 to 1.0.")/100
    acousticness_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='acousticness_weight') / 100
    st.subheader(' ')

# Instrumentalness Checkbox
if instrumentalness_checkbox:
    instrumentalness = st.number_input('Instrumentalness', min_value=0, max_value=100, key='instrumentalness', help="Predicts whether a track contains no vocals. Values closer to 1.0 suggest a higher likelihood of being instrumental. Most songs that are not insturmental have a value less than 0.5, and more near 0.")/100
    instrumentalness_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='instrumentalness_weight') / 100
    st.subheader(' ')

# Valence Checkbox
if valence_checkbox:
    valence = st.number_input('Valence', min_value=0, max_value=100, key='valence', help="A measure of musical positiveness, with 1.0 being the most positive (e.g., happy) and 0.0 being the most negative (e.g., sad).")/100
    valence_weight = st.number_input('Weight (0-100)', min_value=0, max_value=100, key='valence_weight') / 100
    st.subheader(' ')

##calculate button
if st.button('Calculate Recommendations', key='calculate'):
    try:
        if percentile_mode:
            user_preferences = get_user_preferences_percentile(
                artist=artist, artist_weight=artist_weight,
                popularity=popularity, popularity_weight=popularity_weight,
                genre=genre, genre_weight=genre_weight,
                subgenre=subgenre, subgenre_weight=subgenre_weight,
                danceability=danceability, danceability_weight=danceability_weight,
                energy=energy, energy_weight=energy_weight,
                speechiness=speechiness, speechiness_weight=speechiness_weight,
                acousticness=acousticness, acousticness_weight=acousticness_weight,
                instrumentalness=instrumentalness, instrumentalness_weight=instrumentalness_weight,
                valence=valence, valence_weight=valence_weight
            )
        else:
            user_preferences = get_user_preferences(
                artist=artist, artist_weight=artist_weight,
                popularity=popularity, popularity_weight=popularity_weight,
                genre=genre, genre_weight=genre_weight,
                subgenre=subgenre, subgenre_weight=subgenre_weight,
                danceability=danceability, danceability_weight=danceability_weight,
                energy=energy, energy_weight=energy_weight,
                speechiness=speechiness, speechiness_weight=speechiness_weight,
                acousticness=acousticness, acousticness_weight=acousticness_weight,
                instrumentalness=instrumentalness, instrumentalness_weight=instrumentalness_weight,
                valence=valence, valence_weight=valence_weight
            )
        
        # Calculate similarity scores
        spotify_df['similarity_score'] = spotify_df.apply(lambda row: calculate_similarity_score(row, user_preferences), axis=1)
        
        # Sort by similarity score
        recommended_songs = spotify_df.sort_values(by='similarity_score', ascending=False)
        
        # Apply the filter to remove duplicates
        recommended_songs = get_unique_recommendations(recommended_songs)

        # Apply reverse mappings only if genre/subgenre were selected
        recommended_songs['playlist_genre'] = recommended_songs['playlist_genre'].map(reverse_genre_mapping)
        recommended_songs['playlist_subgenre'] = recommended_songs['playlist_subgenre'].map(reverse_subgenre_mapping)
        recommended_songs['track_artist'] = recommended_songs['track_artist'].map(reverse_artist_mapping)

        # Create a copy of the DataFrame with renamed columns for display purposes
        display_df = recommended_songs.rename(columns={
            'track_name': 'Track Name',
            'track_artist': 'Artist',
            'track_popularity': 'Popularity',
            'playlist_genre': 'Genre',
            'playlist_subgenre': 'Subgenre',
            'danceability': 'Danceability',
            'energy': 'Energy',
            'speechiness': 'Speechiness',
            'acousticness': 'Acousticness',
            'instrumentalness': 'Instrumentalness',
            'valence': 'Valence',
            'similarity_score': 'Similarity Score'
        })

        # Multiply all numerical values by 100
        numerical_columns = ['Popularity', 'Danceability', 'Energy', 'Speechiness', 
                             'Acousticness', 'Instrumentalness', 'Valence', 'Similarity Score']
        display_df[numerical_columns] = display_df[numerical_columns] * 100
        
        # Round all numerical values to 2 decimal places
        display_df[numerical_columns] = display_df[numerical_columns].round(2)
        
        # Display the nicely formatted DataFrame
        st.write(f"Top 10 Recommended Songs for You")
        st.dataframe(display_df[['Track Name', 'Artist', 'Popularity', 
                                 'Genre', 'Subgenre', 
                                 'Danceability', 'Energy', 'Speechiness', 
                                 'Acousticness', 'Instrumentalness', 
                                 'Valence', 'Similarity Score']].head(10), hide_index=True)
    except ValueError as e:
        st.error(f"An error occurred: {e}")