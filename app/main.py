from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las orígenes, puedes especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Load data
data = pd.read_csv('./data/interim/data.csv')

# Calculate weighted ratings
def weighted_rating(df):
    df["ratings"] = pd.to_numeric(df["ratings"], errors='coerce')
    df["no_of_ratings"] = pd.to_numeric(df["no_of_ratings"], errors='coerce')
    C = df['ratings'].mean()  # Promedio de todas las calificaciones
    m = df['no_of_ratings'].quantile(0.75)  # Cantidad mínima de calificaciones (percentil 75)

    def bayesian_rating(row):
        v = row['no_of_ratings']
        R = row['ratings']
        return (v / (v + m) * R) + (m / (m + v) * C)

    df = df.copy()
    df['score'] = df.apply(bayesian_rating, axis=1)
    return df.sort_values('score', ascending=False)

popular_products = weighted_rating(data)

# Load precomputed cosine similarity matrix
cosine_sim = np.load('./data/interim/cosine_sim.npy')

@app.get("/top-rated-products")
def get_top_rated_products(n: int = 25):
    top_products = popular_products.head(n)
    return top_products.replace([np.inf, -np.inf], np.nan).dropna().to_dict(orient='records')

@app.get("/product-index/{name}")
def get_product_index(name: str):
    try:
        index = int(data[data['name'] == name].index[0])
        return {"index": index}
    except IndexError:
        raise HTTPException(status_code=404, detail="Product not found")

@app.get("/recommend/{product_index}")
def recommend(product_index: int, n: int = 5):
    if product_index >= len(data):
        raise HTTPException(status_code=404, detail="Product index out of range")
    
    similar_products = list(enumerate(cosine_sim[product_index]))
    similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)[1:n+1]
    recommended_products = data.iloc[[i[0] for i in similar_products]]
    return recommended_products.replace([np.inf, -np.inf], np.nan).dropna().to_dict(orient='records')

