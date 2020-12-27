import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from classes import Dataset, createCluster, cluster_means, create_avg_user, \
    ant_colony_optimization, recommend_init
from classes import predict


class Recommend_cold_start:
    def generate_init(self):
        # verilerin tutulacağı diziler
        user, item, rating, rating_test, test, pcs_matrix, utility = recommend_init()

        n_users = len(user)
        n_items = len(item)

        result = ant_colony_optimization(n_users, pcs_matrix)

        clusterUser = []
        clusterUser = createCluster(result)
        # KNNalgorithm.getKNNalgorithm(clusterUser,1,1,1)

        means = cluster_means(utility, clusterUser)
        user = create_avg_user(user, n_users, utility)

        maximCluster = 382
        utility_copy = np.copy(utility)
        for i in range(0, maximCluster):
            for j in range(0, n_users):
                if utility_copy[i][j] == 0:
                    utility_copy[i][j] = predict(i + 1, j + 1, 2, n_users, pcs_matrix, user, clusterUser, maximCluster)
        print("\rPrediction [User:Rating] = [%d:%d]" % (i, j))

        # test datası ile tehmin arasında MSE
        y_true = []
        y_pred = []
        for i in range(0, n_users):
            for j in range(0, n_items):
                if test[i][j] > 0:
                    y_true.append(test[i][j])
                    y_pred.append(utility[i][j])

        print("Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))

        # 1 çözüm: knn ile mevcut Ant COlony optimization souçlarının eğitim ve test edilmesi. 0,97 mean square error hesapla

        # 2. çözüm user boş olan ratinglerini  o  kullanıcıya ait cluster ortalamalrı ile dolduralım ve klasik pearson yaklaımı ile tekrar başarı hesaplayalım.1,5


if __name__ == '__main__':
    Recommend_cold_start.generate_init(10)
