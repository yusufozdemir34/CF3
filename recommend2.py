import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from RecommendationHelper import Dataset, createCluster, cluster_means, create_avg_user, \
    recommend_init, set_one_for_max_avg_value_others_zero, get_prediction, mean_square_error
from RecommendationHelper import predict
from ant_colony_helper import AntColonyHelper
from ccl import connected_component_labelling


class ColdStartRecommendation:
    def generate_init(self):
        # verilerin tutulacağı diziler
        user, item, test, pcs_matrix, utility, n_users, n_items = recommend_init()

        result = AntColonyHelper.ant_colony_optimization(n_users, pcs_matrix)
        result = set_one_for_max_avg_value_others_zero(result)
        result = connected_component_labelling(result, 4)
        clusterUser = []
        clusterUser = createCluster(result)
        # KNNalgorithm.getKNNalgorithm(clusterUser,1,1,1)

        means = cluster_means(utility, clusterUser)
        user = create_avg_user(user, n_users, utility)

        utility_copy = get_prediction(utility, pcs_matrix, user, clusterUser)

        # test datası ile tehmin arasında MSE
        mean_square_error(test, utility_copy, n_users, n_items)

        # 1 çözüm: knn ile mevcut Ant COlony optimization souçlarının eğitim ve test edilmesi. 0,97 mean square error hesapla

        # 2. çözüm user boş olan ratinglerini  o  kullanıcıya ait cluster ortalamalrı ile dolduralım ve klasik pearson yaklaımı ile tekrar başarı hesaplayalım.1,5


if __name__ == '__main__':
    ColdStartRecommendation.generate_init(10)
