import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from ast import literal_eval

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv("C:/Users/viraj/music_recommendation/data.csv")
df.drop_duplicates()
df.drop_duplicates(subset=['artists', 'duration_ms', 'name'], keep='last')
df['artists'] = df['artists'].map(lambda x: literal_eval(x))
df['artists'] = df['artists'].map(lambda x: x[0])
corr = df.corr().abs()
songs = df[df['year'] > 1980]
songs.reset_index(drop=True, inplace=True)
songs = songs.drop(columns=['id', 'release_date'])  
sound_params = songs[['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']]

metadata = songs.drop(['acousticness', 'danceability','energy', 'instrumentalness', 'key', 
                       'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence'], axis = 1)
scaler = MinMaxScaler()
sound_features = pd.DataFrame()
for col in sound_params.columns:    
    sound_features[col] = scaler.fit_transform(sound_params[col].values.reshape(-1,1)).ravel() 

mdf = metadata.join(sound_features)


def get_index_from_name(name):
    '''
    This function returns the index of the row when given a song name
    '''
    return mdf[mdf["name"]==name].index.tolist()[0]

nn_mdf = mdf.drop(columns=['artists', 'name'])
songs_nn = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(nn_mdf)
distances, indices = songs_nn.kneighbors(nn_mdf)




# # knn
# def recommend_songs(song=None,id=None):
#    if id:
#        for id in indices[id][1:]:
#            print(mdf.iloc[id]["name"])
#    if song:
#        recommendations = []
#        found_id = get_index_from_name(song)
#        for id in indices[found_id][1:]:
#            recommendations.append((mdf.iloc[id]["name"], mdf.iloc[id]["artists"]))
#            print(mdf.iloc[id]["name"], mdf.iloc[id]["artists"])
#        return recommendations

# recommend_songs("Dynamite")













cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10, n_jobs=-1))])
X = nn_mdf.select_dtypes(np.number)
cluster_pipeline.fit(X)
nn_mdf['cluster'] = cluster_pipeline.predict(X)      

nn_mdf = nn_mdf.drop(columns=['duration_ms', 'year', 'popularity'])

songs_nn3 = NearestNeighbors(n_neighbors=11, algorithm='kd_tree', leaf_size = 40, metric = 'euclidean').fit(nn_mdf)

distances3, indices3 = songs_nn3.kneighbors(nn_mdf)













# kmeans 
def recommend_songs_3(song=None):
        recommendations = []
        found_id = get_index_from_name(song)
        for id in indices3[found_id][1:]:
            recommendations.append((mdf.iloc[id]["name"], mdf.iloc[id]["artists"]))
            print(mdf.iloc[id]["name"], mdf.iloc[id]["artists"])
        return recommendations

recommend_songs_3("Dynamite")
# results5 = recommend_songs_3("Life Goes On")











# # song+artist
# def recomend_songs_by_sound_similarity(data, song, artist):
    
#     try: 
#         song_and_artist_data = data[(data['name'] == song) & (data["artists"]== artist)]

#         similar_songs = data.copy()

#         sound_properties = similar_songs.loc[:,['acousticness', 'danceability',
#            'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
#            'speechiness', 'tempo', 'valence']]

#         #Calculate similiary of all songs to the one we want
#         similar_songs['Similarity with song'] = cosine_similarity(sound_properties, sound_properties.to_numpy()[song_and_artist_data.index[0],None]).squeeze()

#         similar_songs.rename(columns={'name': f'Songs Similar to {song}'}, inplace=True)

#         similar_songs = similar_songs.sort_values(by= 'Similarity with song', ascending = False)

#         similar_songs = similar_songs[['artists', f'Songs Similar to {song}',
#           'year','popularity']]

#         similar_songs.reset_index(drop=True, inplace=True)

#         return similar_songs.iloc[1:11]
    
#     except:
#         print("Oops! This song is not included in our dataset")

# recomend_songs_by_sound_similarity(mdf, "Shallow", "Lady Gaga")