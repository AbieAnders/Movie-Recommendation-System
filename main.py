import numpy as np
import pandas as pd
import ast

#Setting the maximum number of columns that are printed(useful for data pre processing)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

movies = pd.read_csv(r'D:\VsCode\Projects\Movie Recommender System\tmdb_5000_movies.csv')
credits = pd.read_csv(r'D:\VsCode\Projects\Movie Recommender System\tmdb_5000_credits.csv')
print(movies.info())
movies = movies.merge(credits, on ='title')

print(movies.shape)
print(credits.shape)

#Remove: *budget, homepage, id, original_language, original_title, *popularity, production_companies, production_countries,
#        *release_date, *revenue, *runtime, *spoken_languages, status, tagline, vote_average, vote_count, movie_id
#Keep: genres, keywords, title, overview, cast, crew, id
movies = movies[['title', 'id', 'genres', 'overview', 'keywords', 'cast', 'crew']]

print(movies.info())

movies.dropna(inplace = True)
print(movies.iloc[0].genres)

#iloc actually returns string values as the key and value pairs over here, so we need to use the 'ast' library

#The convert function converts the data from the dictionaries to obtain only the necessary values and add it into a list
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_top_3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if(counter != 3):
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def convert_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_top_3)
movies['crew'] = movies['crew'].apply(convert_director)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
print(movies.head())

#Remove all spaces to specifically define each identifier and remove confusion for the ml model.
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags'] = movies['genres'] + movies['keywords'] + movies['crew'] + movies['cast']
print(movies.head())

new_df = movies[['id','title','tags']]
print(new_df.head())
#Convert the tags from a group of lists to a space separated string for each row
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
#Set all the tags to lowercase
new_df['tag'] = new_df['tags'].apply(lambda x:x.lower())
print(new_df.head())

#Vectorization plots the tags of each movie onto a 2d space
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 5000, stop_words = 'english')
vectors = cv.fit_transform(new_df['tags']).toarray()
print(cv.get_feature_names_out())

#Stemming
#Stemming involves breaking down all the forms of a word to its base form (Walking, Walked -> Walk)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stemming(text):
    y = []
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

print("----------")
a = stemming("Walking")
print(a)

#new_df['tags'] = new_df['tags'].apply(stemming)
movies.loc[:, 'tags'] = movies['tags'].apply(stemming)
print(cv.get_feature_names_out())

#In higher dimensions, euclidean distance becomes less reliable so we use cosine distance instead to gauge how close vectors are to each other
#Cosine distance is just the angle between the 2 vectors
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
print(cosine_similarity(vectors).shape)

def recommendation(movie):
    try:
        index = new_df[new_df['title'] == movie].index[0]
    except Exception as e:
        print("The movie doesnt exist in the database.") 
        return
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)

recommendation('Bourne')
