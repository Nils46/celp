from helpers import *

# CONTENT BASED FILTERING MSE

df = merge_reviews()
training, test = split_data(df, d = 0.75)

utility = pivot_ratings(training)

df_1 = merge_businesses()
categories_dataframe = extract_genres(df_1)

merge = merge_businesses()
extracted = extract_genres(merge)
pivot = pivot_genres(extracted)
sim_categories = create_similarity_matrix_categories(pivot)

predicted_ratings = predict_ratings(sim_categories, utility, test)
predicted_ratings.dropna()

mse_content = mse(predicted_ratings)
print(mse_content)


# ITEM BASED FILTERING MSE
sim_ratings = create_similarity_matrix_cosine(utility)

predicted_ratings_items = predict_ratings(sim_ratings, utility, test)
mse_item_based = mse(predicted_ratings_items)

print(mse_item_based)


# DATA DENSITY

def number_of_movies(ratings):
 
    return len(ratings['business_id'].unique())

def number_of_users(ratings):

    return len(ratings['user_id'].unique())

def number_of_ratings(ratings):

    return ratings.shape[0]

def rating_density(ratings):

    return number_of_ratings(ratings) / (number_of_movies(ratings) * number_of_users(ratings))

print(rating_density(df))
