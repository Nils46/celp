from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
from helpers import *
import random
    
def recommend(user_id=None, business_id=None, city=None, n=10):

    # business pagina
    if business_id and user_id:

        item_based = cf_item_login(user_id, n, business_id)
        
        if len(item_based) < n:
            missing_amount = n - len(item_based)
            content_based = content_based_filtering_login(user_id, missing_amount)
        
            return item_based + content_based[:missing_amount]
       
        else:
            return item_based
    
    
    #als een bedrijf wordt aangeklikt zonder in te loggen
    if business_id and not user_id:

        item_based = cf_item_logout(business_id, n)
        
        if len(item_based) < n:
            missing_amount = n - len(item_based)
            content_based = content_based_filtering_logout(business_id, missing_amount)
        
            return item_based + content_based[:missing_amount]
       
        else:
            return item_based
    
    
    # homepagina als een gebruiker heeft ingelogt
    if user_id:

        item_based = cf_item_login(user_id, n, business_id)

        if len(item_based) < n:
            missing_amount = n - len(item_based)
            content_based = content_based_filtering_login(user_id, n)
        
            return item_based + content_based[:missing_amount]
       
        else:
            return item_based
        
    
    # geen user ingelogt
    else:
   
        city = random.choice(CITIES)
        random_sample = random.sample(BUSINESSES[city], n)
        return random_sample
