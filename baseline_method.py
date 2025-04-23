import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class BaselineMethod:
    def __init__(self, df, file_path='data_for_kit.csv', features = None):
        self.df = df
        self.file_path = file_path
        self.target = 'ret_crsp'
        self.features = ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open'] if not features else features

    def build(self):
        self.read_csv()
        self.check_missing_values()
        self.data_clustering()

    def read_csv(self):
        self.df = pd.read_csv(self.file_path)   # Read the CSV file
        self.df['date'] = pd.to_datetime(self.df['date']) # Convert date column to datetime
    
    def check_missing_values(self):
        missing_values = self.df.isnull().sum()  # Check for missing values
        if missing_values.sum() == 0:
            print("\nNo missing values found in the dataset!")
        else:
            print(f"\nTotal number of missing values: {missing_values.sum()}")
            #   palceholder for missing value handling
            
    def data_clustering(self):
        # Prepare the data
        x = self.df[self.features]
        y = self.df[self.target]

        # Scale the features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Find optimal number of clusters using elbow method and silhouette score
        inertias = []
        silhouette_scores = []
        k_num = range(2, 11)

        for k in k_num:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(x_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(x_scaled, kmeans.labels_))

        # Plot elbow curve and silhouette scores
        plt.figure(figsize=(12, 5))

        # Elbow curve
        plt.subplot(1, 2, 1)
        plt.plot(k_num, inertias, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')

        # Silhouette score
        plt.subplot(1, 2, 2)
        plt.plot(k_num, silhouette_scores, 'rx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Method')

        plt.tight_layout()
        plt.show()

        # Apply K-means with optimal k (let's say k=3 for this example)
        optimal_k = 3  # Adjust this based on the elbow curve and silhouette score
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(x_scaled)

        # Add cluster labels to the original dataframe
        self.df['Cluster'] = cluster_labels

        # Analyze clusters with respect to the target variable (ret_crsp)
        print("\nCluster Analysis:")
        cluster_analysis = self.df.groupby('Cluster').agg({
            'ret_crsp': ['mean', 'std', 'count'],
            'open_crsp': 'mean',
            'close_crsp': 'mean',
            'log_ret_lag_close_to_open': 'mean'
        }).round(4)

        print(cluster_analysis)

        # Visualize clusters with respect to returns
        plt.figure(figsize=(10, 6))
        for i in range(optimal_k):
            cluster_data = self.df[self.df['Cluster'] == i]
            plt.scatter(cluster_data['close_crsp'],
                       cluster_data['ret_crsp'],
                       label=f'Cluster {i}')

        plt.xlabel('Closing Price')
        plt.ylabel('Returns (ret_crsp)')
        plt.title('Clusters vs Returns')
        plt.legend()
        plt.show()

        # Box plot of returns by cluster
        plt.figure(figsize=(10, 6))
        self.df.boxplot(column='ret_crsp', by='Cluster')
        plt.title('Distribution of Returns by Cluster')
        plt.ylabel('Returns (ret_crsp)')
        plt.show()

        # Print cluster centers
        print("\nCluster Centers (Original Scale):")
        cluster_centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=self.features
        )
        print(cluster_centers)

        # Calculate mean return for each cluster
        print("\nMean Returns by Cluster:")
        print(self.df.groupby('Cluster')['ret_crsp'].mean().round(4))