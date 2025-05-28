import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import scipy.cluster.hierarchy as sch

# Set background image using CSS
background_image_url = "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80"
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
        color: #FF8C00;
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

st.markdown('<h1 class="title">Customer Segmentation Clustering</h1>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Load default dataset if no file is uploaded
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
else:
    try:
        default_path = "D:/DataScienceGenai/2.K-MEANS CLUSTERING/2.K-MEANS CLUSTERING/Mall_Customers.csv"
        dataset = pd.read_csv(default_path)
        st.info("Using default dataset as no file was uploaded.")
    except FileNotFoundError:
        st.error("Default file not found. Please upload a CSV file.")
        st.stop()

# Show dataset
st.subheader('Dataset Preview')
st.write(dataset)

# Select clustering method
clustering_method = st.selectbox("Select Clustering Method", ["K-Means", "Hierarchical", "DBSCAN"])

# Prepare data
X = dataset.iloc[:, [3, 4]].values

if clustering_method == "K-Means":
    st.subheader('K-Means Clustering')

    # Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    st.pyplot(plt)

    # Apply KMeans
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Plot clusters
    plt.figure()
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('K-Means Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    st.pyplot(plt)

elif clustering_method == "Hierarchical":
    st.subheader('Hierarchical Clustering')

    plt.figure()
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    st.pyplot(plt)

    hc = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)

    plt.figure()
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
    plt.title('Hierarchical Clusters of Customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    st.pyplot(plt)

elif clustering_method == "DBSCAN":
    st.subheader('DBSCAN Clustering')

    # Standardize features
    X_scaled = StandardScaler().fit_transform(X)

    # DBSCAN parameters
    eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
    min_samples = st.slider("Minimum samples", min_value=3, max_value=20, value=10, step=1)

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    st.write(f"Estimated number of clusters: **{n_clusters_}**")
    st.write(f"Estimated number of noise points: **{n_noise_}**")

    if len(set(labels)) > 1 and -1 not in labels:
        silhouette = metrics.silhouette_score(X_scaled, labels)
        st.write(f"Silhouette Coefficient: **{silhouette:.3f}**")

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure()
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # black for noise

        class_member_mask = (labels == k)
        xy = X_scaled[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)
    plt.title(f'DBSCAN Clustering (Clusters: {n_clusters_})')
    st.pyplot(plt)

# Signature
st.markdown('<h5 class="neon-text">Made by Ratnaprava</h5>', unsafe_allow_html=True)
