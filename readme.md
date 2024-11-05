# 1. Collaborative based Movie Recomender System
This recommender uses a technique called collaborative filtering, specifically a model-based approach. The core idea of collaborative filtering is that if two users rate or like similar movies, they likely share similar tastes. Therefore, the movies that one user likes can be recommended to the other.

# 2. Data Setup
We start by creating a user-movie rating matrix where:
                                                     -Rows represent movies.
                                                     -Columns represent users.
Each cell in this matrix contains the rating a user gave to a particular movie (or 0 if the user hasn’t rated it).

# 3. Filtering the Matrix
To avoid noise, we filter out:
                             -Movies with too few ratings.
                             -Users who haven’t rated enough movies.
This helps focus only on popular movies and active users, making the recommendations more reliable.