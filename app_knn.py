import streamlit as st
import pickle
import pandas as pd
import requests 
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
file_path = 'anime.csv' 
anime_data = pd.read_csv(file_path)

anime_data['Popularity'] = pd.to_numeric(anime_data['Popularity'], errors='coerce')
anime_data['Popularity'] = anime_data['Popularity'].fillna(anime_data['Popularity'].mean())

anime_data['Score'] = pd.to_numeric(anime_data['Score'], errors='coerce')
anime_data['Score'] = anime_data['Score'].fillna(anime_data['Score'].mean())

anime_data['Members'] = pd.to_numeric(anime_data['Members'], errors='coerce')
anime_data['Members'] = anime_data['Members'].fillna(anime_data['Members'].mean())

anime_data['Episodes'] = pd.to_numeric(anime_data['Episodes'], errors='coerce')
anime_data['Episodes'] = anime_data['Episodes'].fillna(anime_data['Episodes'].mean())

anime_data['Favorites'] = pd.to_numeric(anime_data['Favorites'], errors='coerce')
anime_data['Favorites'] = anime_data['Favorites'].fillna(anime_data['Favorites'].mean())
# Xử lý cột Genres: Tách các thể loại và áp dụng One-Hot Encoding
anime_data['Genres'] = anime_data['Genres'].str.split(', ')
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(anime_data['Genres'])


genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=anime_data.index)
anime_data = pd.concat([anime_data, genres_df], axis=1)
features = anime_data[mlb.classes_.tolist() + ['Score', 'Popularity', 'Members', 'Episodes', 'Favorites']]
scaler = StandardScaler()
features[['Score', 'Popularity', 'Members', 'Episodes', 'Favorites']] = scaler.fit_transform(
    features[['Score', 'Popularity', 'Members', 'Episodes', 'Favorites']])

knn = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean')
knn.fit(features)

# Hàm get_recommendations_with_details
def get_recommendations_with_details(anime_name, n_recommendations=10):
    if anime_name not in anime_data['Name'].values:
        return f"Anime {anime_name} không có trong dữ liệu."
    
    # Tìm chỉ số của phim
    anime_index = anime_data.index[anime_data['Name'] == anime_name].tolist()[0]
    
    # Tìm các phim tương tự
    distances, indices = knn.kneighbors([features.iloc[anime_index]], n_neighbors=n_recommendations + 1)
    
    # Loại bỏ phim gốc khỏi danh sách đề xuất
    similar_indices = indices[0][1:]
    similar_animes = anime_data.iloc[similar_indices]
    
    # Tạo bảng kết quả chi tiết
    original_anime = anime_data.iloc[anime_index]
    
    def get_genres(row):
        return [genre for genre in mlb.classes_ if row[genre] == 1]
    
    original_genres = get_genres(original_anime)
    similar_animes['Genres'] = similar_animes.apply(get_genres, axis=1)
    
    details_df = pd.DataFrame(columns=['Name', 'Score', 'Popularity', 'Members', 'Episodes', 'Favorites', 'Genres'])
    
    for _, row in similar_animes.iterrows():
        details_df = pd.concat([details_df, pd.DataFrame([[row['Name'], row['Score'], row['Popularity'], row['Members'], row['Episodes'], row['Favorites'], ', '.join(row['Genres'])]], columns=['Name', 'Score', 'Popularity', 'Members', 'Episodes', 'Favorites', 'Genres'])])
    
    details_df.reset_index(drop=True, inplace=True)
    
    return details_df
def get_anime_poster(anime_name):
    url = f"https://api.jikan.moe/v4/anime?q={anime_name}&sfw"
    response = requests.get(url)
    data = response.json()
    if "data" in data and isinstance(data["data"], list) and data["data"]:
        first_result = data["data"][0]
        if "images" in first_result and "jpg" in first_result["images"]:
            return first_result["images"]["jpg"]["image_url"]
    return None

# Streamlit
def main():
    anime_title = anime_data['Name'].values
    st.title("Anime Recommendation System")
    selected_movie = st.selectbox('Type or select a movie to get recommendation', anime_title)
    if st.button("Get Recommendations"):
        recommendations = get_recommendations_with_details(selected_movie)
        st.write(f"Selected Movie: {selected_movie}")

        poster_url = get_anime_poster(selected_movie)
        if poster_url:
            st.image(poster_url, width=150)
        else:
            st.write("Poster not found")

        st.write("Similar Movies:")
        recommendations_with_posters = []
        for idx, row in recommendations.iterrows():
            movie = row['Name']
            genre = row['Genres']
            poster_url = get_anime_poster(movie)
            recommendations_with_posters.append((poster_url, movie, genre))

        recommendations_df = pd.DataFrame(recommendations_with_posters, columns=['Poster', 'Anime', 'Genres'])

        for idx, row in recommendations_df.iterrows():
            cols = st.columns([1, 2, 3])
            if row['Poster']:
                cols[0].image(row['Poster'], width=100)
            cols[1].write(row['Anime'])
            cols[2].write(row['Genres'])

if __name__ == "__main__":
    main()
