import dit
import logging
import numpy as np
import pandas as pd
from time import time
from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Queue
from sklearn.cluster import SpectralClustering


########################################################################################################################
#                     DatasetMngr Class                                                                               #
########################################################################################################################
class DatasetMngr(metaclass=ABCMeta):
    """Abstract placeholder for Dataset Manager class. Specific Dataset types should implement each own Manager."""
    def __init__(self, filename):
        self.__filename = filename
        self._cluster_array=[]
        self._clusters_element_list=[]
        self._pdf = None
        self._number_features = 0
        self._adjacency_matrix= None
        pass

    # -------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def load(self):
        pass

    # -------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_mutual_information(self, index1: int = 0, index2: int = 1):
        pass

    # -------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def generate_clusters(self, n_clusters=2, cluster_algorithm="kmeans", random_state=1):
        pass

    # -------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def rank_datasets(self):
        pass

    ####################################################################################################################
    #  Properties
    #-------------------------------------------------------------------------------------------------------------------
    @property
    def filename(self):
        return self.__filename

    @property
    def cluster_array(self):
        return self._cluster_array

    @property
    def cluster_element_list(self):
        return self._clusters_element_list

    @property
    def probability_distribution_function(self):
        return self._pdf

    @property
    def number_features(self):
        return self._number_features

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#


########################################################################################################################
#                     CSVDatasetMngr Class                                                                             #
########################################################################################################################
class DatasetMngrCSV(DatasetMngr):
    """ CSV Dataset Manager: Dataset_Mngr implementation for Comma Separated Values files."""

    # -------------------------------------------------------------------------------------------------------------------
    def __init__(self, filename="", metadata={}, file_has_header=False):
        """Filename: name of the csv file. Headers: either 0 to read or a list with column names."""
        super().__init__(filename)
        self.__df_data = None
        self.__metadata = metadata
        self.__headers = self.__metadata.keys()
        self.__file_has_header = file_has_header
        self.__datasets = []
        self.__np_data = None
        self._number_features = len(list(self.__metadata.keys()))
        self._number_data_samples=None

    # -------------------------------------------------------------------------------------------------------------------
    def load(self):
        """Loads the dataset from the csv file into a data-frame."""
        if self.filename[-4:] != ".csv":
            raise Exception("Incorrect data file. Should be csv.")
        else:
            if self.__file_has_header :
                self.__df_data = pd.read_csv(self.filename, skiprows=1, names=self.__headers)
            else:
                self.__df_data = pd.read_csv(self.filename, header=None, names=list(self.__headers))

            self.__np_data = self.__df_data.to_numpy()
            self._number_data_samples = self.__np_data.shape[0]
        pass

    # -------------------------------------------------------------------------------------------------------------------
    def rank_by_tc(self):
        if self._pdf is None:
            logging.info("{}=> Start PDF Computation.".format(time()))
            self._pdf = dit.inference.distribution_from_data(self.__np_data, 1)
            logging.info("{}=> End PDF Computation.".format(time()))

        all_index = np.arange(len(self.__headers))
        elements = []

        for i in range(len(self.__headers)):
            x_index = np.delete(all_index ,i )
            tgt_index = [i]
            try:
                tc = dit.multivariate.total_correlation(self._pdf, rvs=[x_index, tgt_index])
            except:
                tc = 0
            elements.append( (list(np.array(list(self.__headers))[x_index]),
                              list(np.array(list(self.__headers))[tgt_index]),
                              tc) )

        ranked = sorted(elements, key=lambda x: x[-1] , reverse=True)
        for tup in ranked:
            print("{:3.5f} = {} = {}".format(tup[-1], tup[1], tup[0]))

        print("=="*30)
        for i in [1,2,3,5,10,20]:
            print("For top-{} the performance was {:.3f}.".format(i,self.evaluate(ranked, top_k = i)))

        pass

    # -------------------------------------------------------------------------------------------------------------------
    def evaluate(self, ranked_list: [], target_name: str = "target", top_k: int = 5):
        position = 0
        for ranked_tuple in ranked_list[:top_k]:
            if target_name in ranked_tuple[1]:
                return (1 / top_k) * (top_k - position)
            position += 1

        return 0
        pass

    # -------------------------------------------------------------------------------------------------------------------
    def compute_mutual_information(self, index1: int = 0, index2: int = 1):
        if self._pdf is None:
            logging.info("{}=> Start PDF Computation.".format(time()))
            self._pdf = dit.inference.distribution_from_data(self.__np_data, 1)
            logging.info("{}=> End PDF Computation.".format(time()))

        return dit.shannon.mutual_information(self._pdf, np.array([index1]), np.array([index2]))

    # -------------------------------------------------------------------------------------------------------------------
    def _do_clustering(self, n_clusters, cluster_algorithm):

        if self._adjacency_matrix is None:
            logging.error("{}=> Clustering w/o adj matrix not possible.".format(time()))
            raise Exception("No adj matrix before _do_clustering.")

        logging.info("{}=> Start Clustering. N_Clusters:{}".format(time(), n_clusters))
        clustering = SpectralClustering(n_clusters = n_clusters,
                                        assign_labels = cluster_algorithm,
                                        eigen_solver = "arpack",  # random_state=None, int value for deterministic algo
                                        n_jobs = -1               # -1 for all cpu cores.
                                        )
        clustering.fit(self._adjacency_matrix)
        logging.info("{}=> Done Clustering.".format(time()))

        self._cluster_array = clustering.labels_

        self._clusters_element_list = []
        for i in range(np.max(np.unique(self._cluster_array))+1):
            self._clusters_element_list.append([])

        inputs=list(self.__metadata.keys())

        for i in range(len(self._cluster_array)):
            self._clusters_element_list[self._cluster_array[i]] .append(inputs[i])

        pass

    # -------------------------------------------------------------------------------------------------------------------
    def _optimize_number_clusters_evaluate(self, cluster_algorithm):
        """Optimizes the number of clusters by evaluating the position of the target.
        Not longer in use due to fairness on evaluation.
        To be removed"""
        logging.info("{}=> Start N_Cluster optimization.".format(time()))
        loss_values = np.zeros(self.number_features)
        for num_clusters in range(2, self.number_features):
            self._do_clustering(num_clusters, cluster_algorithm)
            ranked_list = self.rank_datasets()
            loss_values[num_clusters] = 1 - self.evaluate(ranked_list)

        logging.info("{}=> Cluster Optimization Losses: {}.".format(time(), loss_values))
        best_n_cluster = np.where(loss_values == loss_values.min())[-1][-1] + 2
        logging.info("{}=> Done N_Cluster optimization.".format(time()))
        logging.info("{}=> For {} dataset, Best cluster size = {}.".format(time(), self.filename, best_n_cluster))
        self._do_clustering(best_n_cluster, cluster_algorithm)

        return best_n_cluster
        pass

    # -------------------------------------------------------------------------------------------------------------------
    def _compute_silhouette(self):
        """ Computes the silhouette of the cluster. """
        silhouette = np.zeros(self.number_features)
        for vertex_i in range(self.number_features):
            i_cluster = self._cluster_array[vertex_i]
            a_i = 0
            for vertex_j in range(self.number_features):
                if self._cluster_array[vertex_i] == self._cluster_array[vertex_j]:
                    a_i += self._adjacency_matrix[vertex_i][vertex_j]

            size_c_i = np.array(np.where(self._cluster_array==i_cluster))[0].shape[0]       # cluster i belongs

            if size_c_i == 0:
                a_i = 0
            else:
                a_i = a_i / (size_c_i - 1)

            remaining_clusters = np.array([a for a in np.unique(self._cluster_array) if a != i_cluster])

            arr_b_i = np.zeros(remaining_clusters.shape[0])
            for k in range(remaining_clusters.shape[0]):
                size_c_k = np.array(np.where(self._cluster_array == remaining_clusters[k]))[0].shape[0]
                for vertex_j in range(self.number_features):
                    if self._cluster_array[vertex_i] != self._cluster_array[vertex_j]:
                        arr_b_i[k] += self._adjacency_matrix[vertex_i][vertex_j]

                if size_c_k >0:
                    arr_b_i[k] = arr_b_i[k] / size_c_k

            b_i = np.min(arr_b_i)

            if a_i < b_i:
                silhouette[vertex_i] = 1 - (a_i / b_i)
            elif b_i >  a_i :
                silhouette[vertex_i] = (b_i / a_i) - 1

        return np.mean(silhouette)
        pass

    # -------------------------------------------------------------------------------------------------------------------
    def _optimize_number_clusters(self, cluster_algorithm):
        logging.info("{}=> Start N_Cluster optimization.".format(time()))
        loss_values = np.zeros(self.number_features-1)

        for num_clusters in range(2, self.number_features + 1):
            self._do_clustering(num_clusters, cluster_algorithm)
            loss_values[num_clusters-2] = self._compute_silhouette()

        logging.info("{}=> Cluster Optimization Losses: {}.".format(time(), loss_values))
        best_n_cluster = np.where(loss_values == loss_values.max())[-1][-1] +2
        logging.info("{}=> Done N_Cluster optimization.".format(time()))
        logging.info("{}=> For {} dataset, Best cluster size = {}.".format(time(), self.filename, best_n_cluster))
        print("##### Optimal Number of Clusters: {:2d}.-".format(best_n_cluster))

        self._do_clustering(best_n_cluster, cluster_algorithm)

        return best_n_cluster
        pass

    # -------------------------------------------------------------------------------------------------------------------
    def _compute_adjacency_matrix_seq(self):
        """ Computes the adjacency matrix using sequential programming."""

        logging.info("{}=> Starting Adj. Matrix.".format(time()))
        self._adjacency_matrix = []
        for i in range(len(self.__metadata.keys())):
            row =[]
            for j in range(len(self.__metadata.keys())):
                if i == j:
                    row.append(0)
                else:
                    row.append(self.compute_mutual_information(i, j) )
            self._adjacency_matrix.append(row)
        self._adjacency_matrix = np.array(self._adjacency_matrix)

        self._adjacency_matrix = 1 - (self._adjacency_matrix / np.max(self._adjacency_matrix))

        logging.info("{}=> Done Adj. Matrix.".format(time()))
        pass

    # -------------------------------------------------------------------------------------------------------------------
    def _compute_adjacency_matrix(self):
        """ Computes the adjacency matrix using multiprocessing."""
        if self._pdf is None:
            logging.info("{}=> Start PDF Computation.".format(time()))
            self._pdf = dit.inference.distribution_from_data(self.__np_data, 1)
            logging.info("{}=> End PDF Computation.".format(time()))

        self._adjacency_matrix = np.zeros((len(self.__metadata.keys()), len(self.__metadata.keys())))
        logging.info("{}=> Starting Adj. Matrix.".format(time()))

        def compute_cell(q, pdf, i, j):
            """Required to compute the mutual information in parallel."""
            mi = dit.shannon.mutual_information(pdf, np.array([i]), np.array([j]), rv_mode="indices")
            q.put((i, j, mi))

        for i in range(len(self.__metadata.keys())):
            print(".", end="")
            processes = []
            q = Queue()

            for j in range(len(self.__metadata.keys())):
                p = Process(target=compute_cell, args=(q, self._pdf, i, j))
                processes.append(p)
                p.start()

            for p in processes:
                result = q.get()
                self._adjacency_matrix[result[0]][result[1]] = result[2]
                p.join()

        self._adjacency_matrix = 1 - (self._adjacency_matrix / np.max(self._adjacency_matrix))

        logging.info("{}=> Done Adj. Matrix.".format(time()))

        pass

    # -------------------------------------------------------------------------------------------------------------------
    def generate_clusters(self, n_clusters=2, optimize_n_cluster = False, cluster_algorithm = "kmeans"):
        """Computes the clusters using the dataset."""
        if n_clusters > len(self.__metadata.keys()):
            raise ValueError("Number of clusters must be less or equal to {}.".format(len(self.__metadata.keys())))

        # ## COMPUTE ADJ MATRIX
        self._compute_adjacency_matrix()

        # ## DO CLUSTER
        if optimize_n_cluster:
            self._optimize_number_clusters(cluster_algorithm)
        else:
            self._do_clustering(n_clusters, cluster_algorithm)

        pass

    # -------------------------------------------------------------------------------------------------------------------
    def rank_datasets(self):
        logging.info("{}=> Start Ranking.".format(time()))
        if len(self._clusters_element_list) == 0:
            logging.error("{}=> No clusters to rank.".format(time()))
            raise Exception("No clusters found. Run generate_clusters first.")

        ranked = []

        for i in range(np.max(self._cluster_array)+1):
            idx_target = np.where(self._cluster_array == i)[0]
            idx_input = np.where(self._cluster_array != i)[0]

            if len(idx_target) == 0:
                logging.info("{}=> Skipping cluster no. {}. No target elements in cluster".format(time(),i))
                continue

            target_names = self._clusters_element_list[i]
            try:
                cond_entropy = dit.shannon.conditional_entropy(self._pdf, idx_input, idx_target)
            except Exception as e:
                logging.error("{}=> Error Computing Shannon Conditional Entropy: {}.".format(time(),e))
                cond_entropy = np.NINF

            ranked.append((target_names, cond_entropy))

        logging.info("{}=> Done Raking.".format(time()))
        return sorted(ranked, key=lambda ranked: ranked[1], reverse=True)
        pass

    ####################################################################################################################
    #  Properties
    #-------------------------------------------------------------------------------------------------------------------
    @property
    def raw_data(self):
        return self.__df_data

    @property
    def number_data_samples(self):
        return self._number_data_samples
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-#
