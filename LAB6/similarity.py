import math
import pandas as pd
import time

def compute_correlation_similarity(user1: pd.Series, user2: pd.Series) -> float:
    # For every item which user1 and user2 have in common (Set S)
    common_items = user1.index.intersection(user2.index)
        
    user1_avg = user1.mean()
    user2_avg = user2.mean()
    
    dot_product = 0
    user1_norm = 0
    user2_norm = 0
    
    # Common items
    for item in common_items:
        rating_user1 = user1[item]
        rating_user2 = user2[item]
        
        user1_diff = rating_user1 - user1_avg
        user2_diff = rating_user2 - user2_avg 
        
        dot_product += user1_diff * user2_diff 
        user1_norm += user1_diff * user1_diff 
        user2_norm += user2_diff * user2_diff 
        
    denominator = (math.sqrt(user1_norm) * math.sqrt(user2_norm))
    return 0 if denominator == 0 else dot_product / denominator



def compute_cosine_similarity(user1: pd.Series, user2: pd.Series) -> float:
    
    common_items = user1.index.intersection(user2.index)
    user1_norm = 0
    user2_norm = 0
    dot_product = 0
    
    for item in common_items:
        rating_user1 = user1[item]
        rating_user2 = user2[item]

        dot_product += rating_user1 * rating_user2
        user1_norm += rating_user1 * rating_user1
        user2_norm += rating_user2 * rating_user2 
        
    denominator = math.sqrt(user1_norm) * math.sqrt(user2_norm)
    return 0 if denominator == 0 else dot_product / denominator
        
def compute_adjusted_cosine_similarity(user1: pd.Series, user2: pd.Series, item_means: pd.Series) -> float:
    # For every item which user1 and user2 have in common (Set S)
    common_items = user1.index.intersection(user2.index)
    
    dot_product = 0
    user1_norm = 0
    user2_norm = 0
    
    for item in common_items:
        rating_user1 = user1[item]
        rating_user2 = user2[item]
        item_mean = item_means[item]
        
        user1_diff = rating_user1 - item_mean
        user2_diff = rating_user2 - item_mean
        
        dot_product += user1_diff * user2_diff
        user1_norm += user1_diff * user1_diff
        user2_norm += user2_diff * user2_diff
    
    denominator = math.sqrt(user1_norm) * math.sqrt(user2_norm)
    return 0 if denominator == 0 else dot_product / denominator    

if __name__ == "__main__":
    
    item_means = pd.Series([1,2,1,1])
    vector_a, vector_b = pd.Series([1, 3,3,4]), pd.Series([4, 3, 2,1])
    print(compute_cosine_similarity(vector_a, vector_b))
    print(compute_correlation_similarity(vector_a, vector_b))   
    
    print(compute_adjusted_cosine_similarity(vector_a, vector_b, item_means)) 
    


    
    
    

    