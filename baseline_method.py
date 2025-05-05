import pandas as pd
import copulas
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from custom_copulas import SkewedTCopula
from marginal_fitting import fit_marginal_distributions

class BaselineMethod:
    def __init__(self, df, file_path='data_for_kit.csv', features = None):
        self.df = df
        self.file_path = file_path
        self.target = 'ret_crsp'
        self.features = ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open'] if not features else features

    def build(self) -> None:
        self.read_csv()
        self.check_missing_values()

    def read_csv(self) -> None:
        """
        Step 0.1: read the dataset from a CSV file.
        """
        self.df = pd.read_csv(self.file_path)   # Read the CSV file
        self.df['date'] = pd.to_datetime(self.df['date']) # Convert date column to datetime
    
    def check_missing_values(self) -> None:
        """
        Check for missing values in the dataset.
        """
        missing_values = self.df.isnull().sum()
        if missing_values.sum() == 0:
            print("\nNo missing values found in the dataset!")
        else:
            print(f"\nTotal number of missing values: {missing_values.sum()}")
            #palceholder for missing value handling
            
    def data_clustering(self, plot_silhouette=False):
        """
        Method to cluster the data (currently using K-Means clustering).
        :param plot_silhouette: Set to True to plot silhouette scores for different numbers of clusters.
        :return: dataframes for each cluster, and the final clustering results.
        """
        # Select features for clustering
        x = self.df[self.features]
        
        # Standardize the features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        
        # Initialize lists to store silhouette scores
        silhouette_scores = []
        max_clusters = 10
        
        # Calculate silhouette scores for different numbers of clusters
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(x_scaled)
            silhouette_avg = silhouette_score(x_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find optimal number of clusters
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Perform final clustering with optimal number of clusters
        final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        self.df['cluster'] = final_kmeans.fit_predict(x_scaled)
        
        # Create separate dataframes for each cluster
        clustered_dfs = {f'cluster_{i}': self.df[self.df['cluster'] == i] 
                         for i in range(optimal_clusters)}
        
        # Plot silhouette scores if requested
        if plot_silhouette:
            plt.figure(figsize=(10, 6))
            plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
            plt.xlabel('number of clusters')
            plt.ylabel('silhouette score')
            plt.title('silhouette score vs number of clusters')
            plt.grid(True)
            plt.show()
        
        return clustered_dfs

    def fit_copulas(self) -> pd.DataFrame:
        """
        Step 1: Estimate the Marginal Distributions
        :return:
        """
        pass

    def prob_integral_tf(self) -> pd.DataFrame:
        """
        Step 2: Calculate the CDF of all marginals and transform the observed data to uniform random variables
        :return:
        """
        pass

    def fit_copula(self):
        """
        Step 3: Fit a chosen copula to the transformed data
        :return:
        """
        pass

    def evaluate_baseline(self):
        """
        Step 4: Evaluate the model
        :return:
        """

