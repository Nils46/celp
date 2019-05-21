from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
import sklearn.metrics.pairwise as pw 
import numpy as np
import pandas as pd
import random
import operator


def split_data(data, d = 0.75):
    
    np.random.seed(seed=5)
    mask_test = np.random.rand(data.shape[0]) < d
    return data[mask_test], data[~mask_test]
    
    
def mse(predicted_ratings):

    diff = predicted_ratings['stars'] - predicted_ratings['predicted rating']
    return (diff**2).mean()
    
    
def predict_ratings(similarity, utility, to_predict):

    ratings_test_c = to_predict.copy()

    ratings_test_c['predicted rating'] = to_predict.apply(lambda row: predict_ids(similarity, utility, row['user_id'], row['business_id']), axis=1)
    return ratings_test_c
    
    
def predict_ids(similarity, utility, userId, itemId):
    
    if userId in utility.columns and itemId in similarity.index:
        return predict_vectors(utility.loc[:,userId], similarity[itemId])
    return np.nan


def create_similarity_matrix_cosine(matrix):

    mc_matrix = matrix - matrix.mean(axis = 0)
    return pd.DataFrame(pw.cosine_similarity(mc_matrix.fillna(0)), index = matrix.index, columns = matrix.index)

def predict_vectors(user_ratings, similarities):

    relevant_ratings = user_ratings.dropna()

    similarities_s = similarities[relevant_ratings.index]
    
    similarities_s = similarities_s[similarities_s > 0.0]
    relevant_ratings = relevant_ratings[similarities_s.index]
    
    norm = similarities_s.sum()
    if(norm == 0):
        return np.nan
    
    return np.dot(relevant_ratings, similarities_s)/norm


def merge_reviews():

    df = pd.DataFrame()

    for city in CITIES:
        reviews = REVIEWS[city]
        df = df.append(pd.DataFrame.from_dict(reviews))
    
    df = df.drop_duplicates(subset=["business_id", "user_id"], keep="last")
    
    return df
    

def merge_businesses():
    
    df = pd.DataFrame()
    
    for city in CITIES:
        businesses = BUSINESSES[city]
        df = df.append(businesses)
    
    return df


def extract_genres(businesses):

    businesses = businesses.fillna('')
    genres_m = businesses.apply(lambda row: pd.Series([row['business_id']] + row['categories'].lower().split(",")), axis=1)
    stack_genres = genres_m.set_index(0).stack()
    df_stack_genres = stack_genres.to_frame()
    df_stack_genres['business_id'] = stack_genres.index.droplevel(1)
    df_stack_genres.columns = ['categories', 'business_id']
    
    return df_stack_genres.reset_index()[['business_id', 'categories']]


def pivot_genres(df):

    return df.pivot_table(index = 'business_id', columns = 'categories', aggfunc = 'size', fill_value=0)
    
    
def pivot_ratings(df):

    return df.pivot(values='stars', columns='user_id', index='business_id')
    
    
def create_similarity_matrix_categories(matrix):

    npu = matrix.values
    m1 = npu @ npu.T
    diag = np.diag(m1)
    m2 = m1 / diag
    m3 = np.minimum(m2, m2.T)
    return pd.DataFrame(m3, index = matrix.index, columns = matrix.index)
    
    
def reviews(user_id): 
    
    businesses = []

    for city in CITIES:
        for reviews in REVIEWS[city]:
            if reviews["user_id"] == user_id:
                businesses.append(reviews["business_id"])
    
    return businesses


def select_city(user):
    
    result=pd.DataFrame()
    
    review = merge_reviews()
    business = merge_businesses()
    
    data = (review.loc[(review['user_id'] == user)]).business_id
    for x in data:   
        result = result.append(business.loc[(business['business_id'] == x)][['business_id','city']])
    dick={}
    for x in (result.city):
        dick[x]=0
    for x in (result.city):
        dick[x]=dick[x]+1
    dick=sorted(dick.items(), key=lambda x: x[1], reverse=True) 
    
    return(dick[0][0])    
    
    
def select_neighborhood_business(target_business):
        
    merge = merge_businesses()
    extracted = extract_genres(merge)
    pivot = pivot_genres(extracted)
    similarity_matrix = create_similarity_matrix_categories(pivot)
    
    bedrijven = dict(similarity_matrix[similarity_matrix[target_business] > 0][target_business])
    
    return sorted(bedrijven.items(), key=operator.itemgetter(1), reverse=True)
                

def select_neighborhood_user(target_user):
    
    reviewed = reviews(target_user)

    neighbor = {}
    
    bs = merge_businesses().business_id
    
    merge = merge_reviews()
    merge = merge[["business_id","stars","user_id"]]
    pivot = pivot_ratings(merge)

    similarity_matrix = create_similarity_matrix_cosine(pivot)

    for x in reviewed:
        for y in bs:
            if x != y:
                similarity1 = similarity_matrix.loc[y,x]
                if similarity1 > 0 and similarity1 != 1:
                    neighbor.update({y : similarity1})
    
    return(sorted(neighbor.items(), key=operator.itemgetter(1), reverse=True))


def get_business(business_id):
    
    business = merge_businesses()
    
    data = (business.loc[(business['business_id'] == business_id)])
    data_dict = data.to_dict('index')
    
    for x in data_dict:
        return(data_dict[x]) 

    
def content_based_filtering_login(user_id, n):
    
    reviews = reviews(user_id)
    
    result = []
    
    city = select_city(user_id)
    
    for business_id in review:
        neighbors = select_neighborhood_business(business_id)
        
        business_ids = []
    
        for neighbor in neighbors:
            if neighbor[0] != business_id:
                business_ids.append(neighbor[0])
    
        for i_d in business_ids:
            business_data = get_business(i_d)
            if business_data["city"] == city:
                result.append(business_data)
        
    return result[1:n+1]
    
    
def content_based_filtering_logout(business_id, n):
    
    neighbors = select_neighborhood_business(business_id)
    
    business_ids = []
    
    for neighbor in neighbors:
        if neighbor[0] != business_id:
            business_ids.append(neighbor[0])
    
    result = []
    
    for i_d in business_ids:
        business_data = get_business(i_d)
        result.append(uu)
    
    return result[1:n+1]
    

def cf_item_login(user_id, n, business_id):
    
    neighbors = select_neighborhood_user(user_id)
    
    business_ids = []
    
    for neighbor in neighbors:
        if neighbor[0] != business_id:
            business_ids.append(neighbor[0])
    
    result = []
    
    city = select_city(user_id)
    
    for i_d in business_ids:
        business_data = get_business(i_d)
        if business_data["city"] == city:
            result.append(business_data)
        
    return result[1:n+1]
    
def cf_item_logout(business_id, n):
    
    neighbors = select_neighborhood_business(business_id)

    business_ids = []
    
    for neighbor in neighbors:
        if neighbor[0] != business_id:
            business_ids.append(neighbor[0])
    
    result = []
    
    for i_d in business_ids:
        business_data = get_business(i_d)
        result.append(business_data)
        
    return result[1:n+1]
