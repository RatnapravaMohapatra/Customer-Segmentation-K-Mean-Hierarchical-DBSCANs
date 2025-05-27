import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans

# Set background image using CSS
background_image_url = "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"  # Nature background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .title {{
        color: #FF8C00;  /* Dark Orange */
        font-size: 3em;
        text-align: center;
    }}
    .neon-text {{
        color: #39ff14;
        font-size: 1.5em;
        text-align: center;
        text-shadow: 0 0 5px #39ff14, 0 0 10px #39ff14, 0 0 20px #39ff14, 0 0 40px #39ff14;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Title with dark orange color
st.markdown('<h1 class="title">Customer Segmentation K-Means Clustering</h1>', unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("Upload your Mall_Customers.csv file", type='csv')

if uploaded_file is not None:
    # Read dataset
    dataset = pd.read_csv(uploaded_file)
    st.subheader('Dataset Preview')
    st.write(dataset)

    # Prepare data
    X = dataset.iloc[:, [3, 4]].values

    # Calculate WCSS for Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot Elbow Method
    plt.figure()
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    st.pyplot(plt)

    # Training the model
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Plot the clusters
    plt.figure()
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    st.pyplot(plt)

# Add "Made by Ratnaprava" text
st.markdown('<h5 class="neon-text">Made by Ratnaprava</h5>', unsafe_allow_html=True)
