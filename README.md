from sentence_transformers import SentenceTransformer 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

df = pd.read_csv("/Users/arhaann/Documents/code/Python/Netflix_Shows.csv")
ns = pd.read_csv("/Users/arhaann/Documents/code/Python/Netflix_Shows.csv")

st = SentenceTransformer('all-mpnet-base-v2')


df["description"] = df["description"].fillna("")
descriptions = st.encode(df["description"].tolist())
descriptions = pd.DataFrame(descriptions, columns = [f"desc_{i}" for i in range(descriptions.shape[1])])

rate = df["rating"].fillna("")
rate = pd.get_dummies(rate, dummy_na=True)
typ_e = df["type"].fillna("")
typ_e = pd.get_dummies(typ_e, dummy_na=True)
country = df["country"].fillna("")
country = pd.get_dummies(country, dummy_na=True)

cast_vectorizer = TfidfVectorizer(token_pattern=r'[^,]+')
director_vectorizer = TfidfVectorizer(token_pattern=r'[^,]+')
listed_in_vectorizer = TfidfVectorizer(token_pattern=r'[^,]+')

c = cast_vectorizer.fit_transform(ns['cast'].fillna(""))
d = director_vectorizer.fit_transform(ns['director'].fillna(""))
li = listed_in_vectorizer.fit_transform(ns['listed_in'].fillna(""))

c_df = pd.DataFrame(c.toarray(), columns=cast_vectorizer.get_feature_names_out())
d_df = pd.DataFrame(d.toarray(), columns=director_vectorizer.get_feature_names_out())
li_df = pd.DataFrame(li.toarray(), columns=listed_in_vectorizer.get_feature_names_out())

concated = pd.concat([0.2*li_df, 0.1*c_df, 0.05*d_df, 0.65*descriptions, 0.05*country, 0.05*rate, 0.05*typ_e], axis=1)


X = normalize(concated.values, norm = 'l2')

sim_matrix = cosine_similarity(X)
def get_input():
    while True:
        inp = input("Enter show/movie name(Be cautious with grammer): ")
        exists = inp in df['title'].values
        if exists == False:
            print("Output Invalid! ")
            continue
        else:
            return inp

def rec(inp, n):
    idx = df.index[df["title"] == inp][0]
    scores = sim_matrix[idx]
    sim_index = scores.argsort()[::-1]
    sim_index = sim_index[1:]
    top_n = sim_index[:n]
    titles = df['title'].iloc[top_n].values
    return titles

get_inp = get_input()
print(get_inp)
print(rec(get_inp, 5))
