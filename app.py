from h2o_wave import main, app, Q, ui
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies = pd.read_csv("E:\\My Office Work\\IXD Labs\\Workspace\\Movie Recomender - Content Based Filtering\\dataset\\movies.csv")
ratings = pd.read_csv("E:\\My Office Work\\IXD Labs\\Workspace\\Movie Recomender - Content Based Filtering\\dataset\\ratings.csv")

final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Convert to sparse matrix format
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Fit the NearestNeighbors model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Recommendation function without similarity scores
def get_recommendation(movie_name):
    movie_list = movies[movies['title'].str.contains(movie_name, case=False)]
    
    if not movie_list.empty:
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]

        # Get distances and indices
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=11)
        recommended_movies = []
        
        for i in range(1, len(indices.flatten())):
            idx = final_dataset.iloc[indices.flatten()[i]]['movieId']
            movie_title = movies[movies['movieId'] == idx]['title'].values[0]
            recommended_movies.append(movie_title)
        
        return recommended_movies
    else:
        return ["Movie not found!"]

# H2O Wave app
@app('/recommend')
async def serve(q: Q):
    if not q.client.initialized:
        q.page['header'] = ui.header_card(
            box='1 1 4 1',
            title='Movie Recommendation System',
            subtitle='Enter a movie name to get recommendations.',
        )

        q.page['input'] = ui.form_card(
            box='1 2 4 2',
            items=[
                ui.textbox(name='movie_name', label='Enter Movie Name Here', placeholder='Type a movie name...'),
                ui.button(name='recommend', label='Get Recommendations', primary=True)
            ]
        )

        q.page['output'] = ui.markdown_card(
            box='1 4 4 4',
            title='Recommendations',
            content='Your recommendations will appear here.'
        )

        q.client.initialized = True

    if q.args.recommend:
        movie_name = q.args.movie_name
        if movie_name:
            recommendations = get_recommendation(movie_name)
            if recommendations:

                q.page['output'].content = '\n'.join(f"- {movie}" for movie in recommendations)
            else:
                q.page['output'].content = 'No recommendations found.'
        else:
            q.page['output'].content = 'Please enter a movie name.'

    await q.page.save()