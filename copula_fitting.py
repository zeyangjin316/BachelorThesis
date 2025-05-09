import datetime
import pandas as pd
import matplotlib.pyplot as plt
from model import CustomModel
from copulas.multivariate import GaussianMultivariate
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class CopulaEstimator(CustomModel):
    def __init__(self, df, split_point: float|datetime =0.8, file_path: str = 'data_for_kit.csv',
                 features: list[str] = None):
        """
        Initialize the BaselineMethod experiment.
        
        Args:
            df:             Input DataFrame
            split_point:    Either a float between 0 and 1 representing the percentage of data for training,
                            or a datetime-like object specifying the split date. Default is 0.8 (80% training).
            file_path:      Path to the CSV file. Default is 'data_for_kit.csv'.
            features:       List of feature columns to use. Default is ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open'].
        """
        super().__init__(df, split_point)
        self.split_point = split_point
        self.file_path = file_path
        self.target = 'ret_crsp'
        self.features = ['open_crsp', 'close_crsp', 'log_ret_lag_close_to_open'] if not features else features

        self.fitted_copula = None
        self.fitted_marginals = None

    def run(self):
        self.train()
        self.evaluate()
        return self.fitted_copula, self.fitted_marginals

    def train(self):
        marginals = self._transform_train_data(self.train_set)
        self.fitted_copula = self._fit_copula(marginals)

    def test(self):
        pass

    def predict(self):
        pass

    def evaluate(self, true_values, predicted_values):
        pass

    def _data_clustering(self, plot_silhouette=False):
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

    def _transform_train_data(self, data: pd.DataFrame) -> dict[str, object]:
        """
        Fits marginal distributions to the provided data and transforms the data using
        Probability Integral Transform (PIT) to uniform distribution.

        :param data: The input data as a pandas DataFrame to fit and transform.
        :type data: pd.DataFrame

        :return: Dictionary containing transformed data mapped to uniform distribution.
        :rtype: dict[str, object]
        """
        from copula_marginals import fit_marginal_distributions, transform_to_uniform

        # First fit the distributions
        self.fitted_marginals = fit_marginal_distributions(data)

        # Then transform to uniform using PIT
        uniform_data = transform_to_uniform(data, self.fitted_marginals)

        return uniform_data

    def _fit_copula(self, marginals: dict[str, object]):
        """
        Step 3: Fit a Gaussian copula to the transformed data
        
        Args:
            marginals: Dictionary containing the PIT (Probability Integral Transform) values
                  for each symbol's returns
        
        Returns:
            fitted_copula: The fitted Gaussian copula object
        """

        # Convert the marginals dictionary to a DataFrame
        # Each column will be the PIT values for a symbol
        uniform_data = pd.DataFrame(marginals)

        # Initialize and fit the Gaussian copula
        gaussian_copula = GaussianMultivariate()
        gaussian_copula.fit(uniform_data)

        return gaussian_copula