import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from classes import Dataset
from classes import pearson
from classes import predict
from classes import isaverage
from ccl import connected_component_labelling
from antgraph import AntGraph
import antcolony
from scipy.stats import pearsonr
# verilerin tutulacağı diziler
user = []
item = []

rating = []
rating_test = []

# Dataset class kullanarak veriyi dizilere aktarma
d = Dataset()
d.load_users("data/u.user", user)
d.load_items("data/u.item", item)
d.load_ratings("data/ua.base", rating)
d.load_ratings("data/ua.test", rating_test)

n_users = len(user)
n_items = len(item)
n_users

# utility user-item tablo sonucu olarak rating tutmaktadır.
# NumPy sıfırlar işlevi, yalnızca sıfır içeren NumPy dizileri oluşturmanıza olanak sağlar.
# Daha da önemlisi, bu işlev dizinin tam boyutlarını belirlemenizi sağlar.
# Ayrıca tam veri türünü belirlemenize de olanak tanır.
utility = np.zeros((n_users, n_items))
for r in rating:
    utility[r.user_id - 1][r.item_id - 1] = r.rating

# print(utility)

test = np.zeros((n_users, n_items))
for r in rating_test:
    test[r.user_id - 1][r.item_id - 1] = r.rating

# Itemların genre üzerindeki clusterı
movie_genre = []
for movie in item:
    movie_genre.append([movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
                        movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
                        movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war,
                        movie.western])

movie_genre = np.array(movie_genre)

print(movie_genre.size)
print(movie_genre.ndim)

# clusteri kaldirdiğimizda ortalamayı nasıl bulup ekleyeceğiz.
# prediction daki [cluster.labels_[j] yerine ne ekleycegiz
cluster = KMeans(n_clusters=19)
cluster.fit_predict(movie_genre)
# modell uygulanması.


utility_clustered = []

for i in range(0, n_users):
    average = np.zeros(19)
    tmp = []
    for m in range(0, 19):
        tmp.append([])
        # n. kullanıcı için [0-19]'a kadar cluster için verilen oylar
    for j in range(0, n_items):
        if utility[i][j] != 0:
            tmp[cluster.labels_[j] - 1].append(utility[i][j])
            # her tür clusterı için verilen oylar tmpde
    for m in range(0, 19):
        if len(tmp[m]) != 0:
            average[m] = np.mean(tmp[m])
            # her tür clusterı için verilen oyların ortalamaları
        else:
            average[m] = 0
    utility_clustered.append(average)
# her userın clusterlara verdiği oy ortalaması
utility_clustered = np.array(utility_clustered)

# her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
for i in range(0, n_users):
    x = utility_clustered[i]
    user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)

pcs_matrix = np.zeros((n_users, n_users))
for i in range(0, n_users):
    for j in range(0, i):
        if i != j:
            pcs_matrix[i][j] = pearson(i + 1, j + 1, utility_clustered, user)

print("\rSimilarity Matrix [%d:%d] = %f" % (i + 1, j + 1, pcs_matrix[i][j]))

# print(pcs_matrix)
graph = AntGraph(n_users, pcs_matrix)
graph.reset_tau()
num_iterations = 5
# n_users = 5
ant_colony = antcolony.AntColony(graph, 5, num_iterations)
ant_colony.start()
graph.delta_mat = isaverage(graph.delta_mat)
result = connected_component_labelling(graph.delta_mat, 4)

clusterNumber = result.max()
clusterUser = []
for m in range(0, clusterNumber):
    clusterUser.append([])
for i in range(0, len(result)):
    for j in range(0, 3):  # len(result[i])):
        clusterUser[result(i, j) - 1].append(i)

utility_copy = np.copy(utility_clustered)
for i in range(0, n_users):
    for j in range(0, 19):
        if utility_copy[i][j] == 0:
            utility_copy[i][j] = predict(i + 1, j + 1, 50, n_users, result, user, utility_clustered)
# print("\rPrediction [User:Rating] = [%d:%d]" % (i, j))

print(utility_copy)

# test datası ile tehmin arasında MSE
y_true = []
y_pred = []
for i in range(0, n_users):
    for j in range(0, n_items):
        if test[i][j] > 0:
            y_true.append(test[i][j])
            y_pred.append(utility_copy[i][cluster.labels_[j] - 1])

print("Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))
