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

# clusteri kaldirdiğimizda ortalamayı nasıl bulup ekleyeceğiz.
# prediction daki [cluster.labels_[j] yerine ne ekleycegiz
pcs_matrix = np.zeros((n_users, n_users))

for i in range(0, n_users):
    for j in range(0, i):
        if i != j:
            A = utility[i]
            B = utility[j]
            pcs_matrix[i][j], _ = pearsonr(A, B)
            # pcs_matrix[i][j] = pearson(i + 1, j + 1, utility, user)
# print(pcs_matrix)
graph = AntGraph(n_users, utility)
graph.reset_tau()
num_iterations = 5
# n_users = 5
ant_colony = antcolony.AntColony(graph, 5, num_iterations)
ant_colony.start()
graph.delta_mat = isaverage(graph.delta_mat)
result = connected_component_labelling(graph.delta_mat, 4)



print("\rSimilarity Matrix [%d:%d] = %f" % (i + 1, j + 1, pcs_matrix[i][j]))

utility_copy = np.copy(utility)
for i in range(0, n_users):
    for j in range(0, n_items):
        if utility_copy[i][j] == 0:
            utility_copy[i][j] = predict(i + 1, j + 1, 50, n_users, result, user, utility)
# print("\rPrediction [User:Rating] = [%d:%d]" % (i, j))


# test datası ile tehmin arasında MSE
y_true = []
y_pred = []
for i in range(0, n_users):
    for j in range(0, n_items):
        if test[i][j] > 0:
            y_true.append(test[i][j])
            y_pred.append(utility[i][j])

print("Mean Squared Error: %f" % mean_squared_error(y_true, y_pred))
