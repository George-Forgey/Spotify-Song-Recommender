<!-- Improved compatibility of back to top link -->
<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO & TITLE -->
<br />
<div align="center">
  <img src="https://images.unsplash.com/photo-1495434942214-9b525bba74e9?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80" alt="Spotify Recommender" width="400">

  <h1 align="center">🎧 Spotify Song Recommender 🎶</h1>
  
  <p align="center">
    Input your song preferences and receive personalized Spotify song recommendations!
    <br />
    <a href="https://spotify-song-recommender-bot.streamlit.app/"><strong>🌐 Try the App »</strong></a>
    <br />
    <br />
    <a href="https://github.com/George-Forgey/spotify-song-recommender">Report Bug</a>
    ·
    <a href="https://github.com/George-Forgey/spotify-song-recommender">Request Feature</a>
  </p>
</div>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>📜 Table of Contents</summary>
  <ol>
    <li><a href="#project-overview">Project Overview</a></li>
    <li><a href="#data-sources">Data Sources</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#technologies-used">Technologies Used</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#inspiration-and-thanks">Inspiration and Thanks</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

<!-- PROJECT OVERVIEW -->
## Project Overview

**Spotify Song Recommender** helps music enthusiasts discover new songs based on their existing preferences. Simply choose a track you like, and this recommender will suggest similar tunes you might enjoy. It leverages the powerful [`spotifyr` package](https://www.rcharlie.com/spotifyr/) to access detailed audio features and metadata directly from Spotify’s API.

This project aims to make music discovery seamless, fun, and deeply personalized. By analyzing various audio attributes (like danceability, valence, energy, and more), we’re able to recommend tracks that resonate with your tastes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- DATA SOURCES -->
## Data Sources

The data comes from Spotify via the [`spotifyr` package](https://www.rcharlie.com/spotifyr/). [Charlie Thompson](https://twitter.com/_RCharlie), [Josiah Parry](https://twitter.com/JosiahParry), Donal Phipps, and Tom Wolff authored this package to streamline retrieving both user-specific and general metadata around tracks from Spotify’s API.

Make sure to explore the [`spotifyr` package website](https://www.rcharlie.com/spotifyr/) to learn how you can collect your own data!

Additionally, [Kaylin Pavlik](https://twitter.com/kaylinquest/status/1213138536570015745) wrote a [blog post](https://www.kaylinpavlik.com/classifying-songs-genres/) using Spotify’s audio features to classify songs by genre. She collected about 5000 songs from 6 main categories (EDM, Latin, Pop, R&B, Rap, & Rock) using `spotifyr`.

**Special Mentions**:
- [Jon Harmon](https://github.com/rfordatascience/tidytuesday/issues/160)
- [Neal Grantham](https://twitter.com/nsgrantham/status/1213190975113199616)

The collective work of these individuals inspired and informed the data collection and analysis techniques used here.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- FEATURES -->
## Features

- 🎵 **Personalized Recommendations**: Suggests tracks similar to your chosen song.
- ⚡ **Rich Audio Features**: Analyzes multiple attributes (energy, danceability, tempo, etc.) to ensure top-tier recommendations.
- 🎧 **Interactive Web App**: User-friendly interface built with Streamlit for instant results.
- 🔄 **Continuously Updating**: Leverages the Spotify API for up-to-date song data.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- TECHNOLOGIES USED -->
## Technologies Used

- **Python 3.8+**: Core language for data analysis and application logic.
- **Streamlit**: For creating an interactive, web-based user interface.
- **Spotifyr**: Streamlined access to the Spotify API and its song metadata.
- **Pandas**: Data manipulation and cleaning.
- **NumPy**: Mathematical operations and vectorized computations.
- **Matplotlib/Seaborn**: Visualizing distributions and trends (if used).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- USAGE -->
## Usage

1. **Run the Web App**:  
   This recommender is deployed at:  
   [https://spotify-song-recommender-bot.streamlit.app/](https://spotify-song-recommender-bot.streamlit.app/)

   Just visit the link, enter a song you love, and let the app handle the rest!

2. **Local Setup (Optional)**:
   - Clone the repository:
     ```bash
     git clone https://github.com/your-username/spotify-song-recommender.git
     cd spotify-song-recommender
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run locally:
     ```bash
     streamlit run app.py
     ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- INSPIRATION AND THANKS -->
## Inspiration and Thanks

This project is inspired by efforts to make music discovery more intuitive. By combining open-source tools and insights from data enthusiasts, we push the boundaries of personalized recommendations.

Shout out to the data community and all contributors who developed and documented `spotifyr`, making Spotify data more accessible.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- LICENSE -->
## License

Distributed under the **MIT License**. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- CONTACT -->
## Contact

**Project Maintainer**: [George Forgey](https://github.com/George-Forgey)  
**Email**: [forgey.g@northeastern.edu](mailto:forgey.g@northeastern.edu)  
**LinkedIn**: [My LinkedIn Profile](https://linkedin.com/in/george-forgey)

Project Link: [https://github.com/George-Forgey/spotify-song-recommender](https://github.com/George-Forgey/spotify-song-recommender)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Data Dictionary


|variable                 |class     |description |
|:---|:---|:-----------|
|track_id                 |character | Song unique ID|
|track_name               |character | Song Name|
|track_artist             |character | Song Artist|
|track_popularity         |double    | Song Popularity (0-100) where higher is better |
|track_album_id           |character | Album unique ID|
|track_album_name         |character | Song album name |
|track_album_release_date |character | Date when album released |
|playlist_name            |character | Name of playlist |
|playlist_id              |character | Playlist ID|
|playlist_genre           |character | Playlist genre |
|playlist_subgenre        |character | Playlist subgenre|
|danceability             |double    | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. |
|energy                   |double    | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. |
|key                      |double    | The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. |
|loudness                 |double    | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.|
|mode                     |double    | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.|
|speechiness              |double    | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. |
|acousticness             |double    | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.|
|instrumentalness         |double    | Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. |
|liveness                 |double    | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. |
|valence                  |double    | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). |
|tempo                    |double    | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. |
|duration_ms              |double    | Duration of song in milliseconds |

---

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/George-Forgey/spotify-song-recommender.svg?style=for-the-badge
[contributors-url]: https://github.com/George-Forgey/spotify-song-recommender/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/George-Forgey/spotify-song-recommender.svg?style=for-the-badge
[forks-url]: https://github.com/George-Forgey/spotify-song-recommender/network/members
[stars-shield]: https://img.shields.io/github/stars/George-Forgey/spotify-song-recommender.svg?style=for-the-badge
[stars-url]: https://github.com/George-Forgey/spotify-song-recommender/stargazers
[issues-shield]: https://img.shields.io/github/issues/George-Forgey/spotify-song-recommender.svg?style=for-the-badge
[issues-url]: https://github.com/George-Forgey/spotify-song-recommender/issues
[license-shield]: https://img.shields.io/github/license/George-Forgey/spotify-song-recommender.svg?style=for-the-badge
[license-url]: https://github.com/George-Forgey/spotify-song-recommender/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/george-forgey
