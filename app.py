# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:05:33 2024

@author: SHIVI
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("cosmetics.csv")

df = load_data()

# Sidebar
st.sidebar.title("Cosmetic Analysis")
analysis_option = st.sidebar.selectbox(
    "Choose analysis type:",
    ["T-SNE Plot", "Top Ingredients", "Skin Type Recommendations"]
)

# Display data preview
st.subheader("Dataset Preview")
st.dataframe(df.sample(5))

# Option 1: T-SNE Plot
if analysis_option == "T-SNE Plot":
    # Filter for moisturizers for dry skin
    moisturizers = df[df.Label == "Moisturizer"]
    moisturizers_dry = moisturizers[moisturizers.Dry == 1].reset_index(drop=True)

    # Tokenize ingredients
    st.write("Processing data for t-SNE...")
    ingredient_idx = {}
    corpus = []
    idx = 0

    for i in range(len(moisturizers_dry)):
        ingredients = moisturizers_dry["Ingredients"][i].lower().split(", ")
        corpus.append(ingredients)
        for ingredient in ingredients:
            if ingredient not in ingredient_idx:
                ingredient_idx[ingredient] = idx
                idx += 1

    N = len(ingredient_idx)
    M = len(moisturizers_dry)
    A = np.zeros((M, N))

    def oh_encoder(tokens):
        x = np.zeros(N)
        for ingredient in tokens:
            x[ingredient_idx[ingredient]] = 1
        return x

    for i, tokens in enumerate(corpus):
        A[i, :] = oh_encoder(tokens)

    # Apply t-SNE
    model = TSNE(n_components=2, learning_rate=200, random_state=42)
    tsne_features = model.fit_transform(A)
    moisturizers_dry["X"] = tsne_features[:, 0]
    moisturizers_dry["Y"] = tsne_features[:, 1]

    # Create Bokeh plot
    st.write("Generating t-SNE visualization...")
    source = ColumnDataSource(moisturizers_dry)
    plot = figure(
        title="T-SNE Visualization of Cosmetics",
        x_axis_label="T-SNE 1",
        y_axis_label="T-SNE 2",
        width=700,
        height=500,
    )
    plot.circle(x="X", y="Y", source=source, size=10, color="#FF7373", alpha=0.8)
    hover = HoverTool(
        tooltips=[("Item", "@Name"), ("Brand", "@Brand"), ("Price", "$@Price"), ("Rank", "@Rank")]
    )
    plot.add_tools(hover)

    st.bokeh_chart(plot)

# Option 2: Top Ingredients
elif analysis_option == "Top Ingredients":
    # Find most common ingredients
    ingredient_counts = Counter(df["Ingredients"].str.split(",").sum())
    common_ingredients = ingredient_counts.most_common(10)
    ingredients_df = pd.DataFrame(common_ingredients, columns=["Ingredient", "Count"])

    # Plot top ingredients
    st.write("Top 10 Most Common Ingredients")
    plt.figure(figsize=(10, 7))
    sns.barplot(x="Count", y="Ingredient", data=ingredients_df)
    plt.title("Top 10 Most Common Ingredients")
    st.pyplot(plt)

# Option 3: Skin Type Recommendations with Label Filter
elif analysis_option == "Skin Type Recommendations":
    # Add a dropdown for skin type selection
    skin_type = st.sidebar.selectbox(
        "Select your skin type:",
        ["Dry", "Oily", "Combination", "Sensitive", "Normal"]
    )

    # Add a dropdown for label selection (Moisturizer, Serum, etc.)
    label_type = st.sidebar.selectbox(
        "Select product label:",
        ["Moisturizer", "Serum", "Cleanser", "Sunscreen", "Toner"]  # Update with relevant labels in your data
    )

    # Filter products based on selected skin type
    if skin_type == "Dry":
        recommended_products = df[df.Dry == 1]
    elif skin_type == "Oily":
        recommended_products = df[df.Oily == 1]
    elif skin_type == "Combination":
        recommended_products = df[df.Combination == 1]
    elif skin_type == "Sensitive":
        recommended_products = df[df.Sensitive == 1]
    else:
        recommended_products = df[df.Normal == 1]

    # Further filter products based on selected label
    recommended_products = recommended_products[recommended_products.Label == label_type]

    # Sort by rank (descending order)
    recommended_products = recommended_products.sort_values(by="Rank", ascending=False)

    # Display the recommended products
    st.write(f"Recommended {skin_type} Skin Products with Label '{label_type}' (Sorted by Rank)")
    st.dataframe(recommended_products[['Name', 'Brand', 'Price', 'Rank']])

# Footer
st.sidebar.write("Built with ❤️ using Streamlit")  