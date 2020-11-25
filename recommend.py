import sys
import time
import math
import re
import pickle

import numpy as np
import antcolony
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from antgraph import AntGraph
import numpy as np
from PIL import Image
from union_find import UnionFind
from ccl import connected_component_labelling

# verilerin tutulacağı diziler
user = []
item = []

rating = []
rating_test = []
d = []
n_users = 0
n_items = 0
cluster = []
utility_clustered = []
graph = set()
pcs_matrix = []
test = []


class User:
    def __init__(self, id, age, sex, occupation, zip):
        self.id = int(id)
        self.age = int(age)
        self.sex = sex
        self.occupation = occupation
        self.zip = zip
        self.avg_r = 0.0


class Item:
    def __init__(self, id, title, release_date, video_release_date, imdb_url, \
                 unknown, action, adventure, animation, childrens, comedy, crime, documentary, \
                 drama, fantasy, film_noir, horror, musical, mystery, romance, sci_fi, thriller, war, western):
        self.id = int(id)
        self.title = title
        self.release_date = release_date
        self.video_release_date = video_release_date
        self.imdb_url = imdb_url
        self.unknown = int(unknown)
        self.action = int(action)
        self.adventure = int(adventure)
        self.animation = int(animation)
        self.childrens = int(childrens)
        self.comedy = int(comedy)
        self.crime = int(crime)
        self.documentary = int(documentary)
        self.drama = int(drama)
        self.fantasy = int(fantasy)
        self.film_noir = int(film_noir)
        self.horror = int(horror)
        self.musical = int(musical)
        self.mystery = int(mystery)
        self.romance = int(romance)
        self.sci_fi = int(sci_fi)
        self.thriller = int(thriller)
        self.war = int(war)
        self.western = int(western)


class Rating:
    def __init__(self, user_id, item_id, rating, time):
        self.user_id = int(user_id)
        self.item_id = int(item_id)
        self.rating = int(rating)
        self.time = time


# User - Item ve Rating verilerini belirlenecek dizilere eklemeyi sağlayacak.
class Dataset:
    def load_users(self, file, u):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 5)
            if len(e) == 5:
                u.append(User(e[0], e[1], e[2], e[3], e[4]))
        f.close()

    def load_items(self, file, i):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('|', 24)
            if len(e) == 24:
                i.append(Item(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], \
                              e[11], e[12], e[13], e[14], e[15], e[16], e[17], e[18], e[19], e[20], e[21], \
                              e[22], e[23]))
        f.close()

    def load_ratings(self, file, r):
        f = open(file, "r")
        text = f.read()
        entries = re.split("\n+", text)
        for entry in entries:
            e = entry.split('\t', 4)
            if len(e) == 4:
                r.append(Rating(e[0], e[1], e[2], e[3]))
        f.close()


# Pearson Korelasyonu. Userlar arasında dolayısı ile user based.
# item based olması için itemler arasında ilişki hesabı da yapılacak.
def pearson(x, y):
    num = 0
    den1 = 0
    den2 = 0
    A = utility_clustered[x - 1]
    B = utility_clustered[y - 1]
    num = sum((a - user[x - 1].avg_r) * (b - user[y - 1].avg_r) for a, b in zip(A, B) if a > 0 and b > 0)
    den1 = sum((a - user[x - 1].avg_r) ** 2 for a in A if a > 0)
    den2 = sum((b - user[y - 1].avg_r) ** 2 for b in B if b > 0)
    den = (den1 ** 0.5) * (den2 ** 0.5)
    if den == 0:
        return 0
    else:
        return num / den


# feromon matrisini ant_colony değerinden dönüş olarak alacağız.
# feromon matrisinin ortalamanın altındakileri sıfır üstündekileri 1 yapıyoruz.
# Yeni feromon matrisini ccl.py deki cluster koduna yazalım.
# Kmeans kısmını atlayıp arkasındaki koda yeni matrisi parametre olarak vereceğiz.
def isaverage(delta_mat):
    avg = delta_mat.mean()
    for i in range(0, np.size(delta_mat, 0)):
        for j in range(0, np.size(delta_mat, 1)):
            if delta_mat[i][j] > avg:
                delta_mat[i][j] = 1
            else:
                delta_mat[i][j] = 0
    return delta_mat


def norm():
    normalize = np.zeros((n_users, 19))
    for i in range(0, n_users):
        for j in range(0, 19):
            if utility_clustered[i][j] != 0:
                normalize[i][j] = utility_clustered[i][j] - user[i].avg_r
            else:
                normalize[i][j] = float('Inf')
    return normalize


# user_id - oyu tahmin edilecek user
# i_id - kullanıcının tahmin edilecek oyu verdiği item clusterı
# top_n - bu benzerlik hesabı için kullanılacak benzer user sayısı.
def predict(user_id, i_id, top_n):
    similarity = []
    for i in range(0, n_users):
        if i + 1 != user_id:
            similarity.append(pcs_matrix[user_id - 1][i])
    temp = norm()
    temp = np.delete(temp, user_id - 1, 0)
    top = [x for (y, x) in sorted(zip(similarity, temp), key=lambda pair: pair[0], reverse=True)]
    # top: benzerlik ve oylama matrislerinin zip ile eşleşmesi sonucu sorted ile sıralanması ile
    # en yüksek benzerlik oranına sahip bireylerin oylarını saklar.
    s = 0
    c = 0
    for i in range(0, top_n):
        if top[i][i_id - 1] != float(
                'Inf'):  # infinitive : sınırsız bir üst değer işlevi görür. bu işin sonuna kadar yani
            s += top[i][i_id - 1]  # top'daki oyların toplamı
            c += 1  # oy sayısı. bu hem ortalama için hem de oy olup olmadığı kontrolü için
    rate = user[user_id - 1].avg_r if c == 0 else s / float(c) + user[user_id - 1].avg_r
    # eğer hiç oy yoksa kullanıcının kendi ortalama oyunu kabul et
    # oy varsa en benzer kullanıcıların o film için verdiği oyların ortalamasını kullanıcı için ata. USER-BASED
    if rate < 1.0:
        return 1.0
    elif rate > 5.0:
        return 5.0
    else:
        return rate


def findresult(utility_clustered1):
    utility_copy = np.copy(utility_clustered1)
    for i in range(0, n_users):
        for j in range(0, 19):
            if utility_copy[i][j] == 0:
                # sys.stdout.write("\rPrediction [User:Rating] = [%d:%d]" % (i, j))
                # print ile tüm döngüyü yazıyor. stdout ile her sonuç dönüyor.
                utility_copy[i][j] = predict(i + 1, j + 1, 50)
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


def run():
    # Dataset class kullanarak veriyi dizilere aktarma
    d = Dataset()
    d.load_users("data/u.user", user)
    d.load_items("data/u.item", item)
    d.load_ratings("data/ua.base", rating)
    d.load_ratings("data/ua.test", rating_test)

    n_users = len(user)
    n_items = len(item)
    print(n_users)
    print(n_items)

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
        movie_genre.append(
            [movie.unknown, movie.action, movie.adventure, movie.animation, movie.childrens, movie.comedy,
             movie.crime, movie.documentary, movie.drama, movie.fantasy, movie.film_noir, movie.horror,
             movie.musical, movie.mystery, movie.romance, movie.sci_fi, movie.thriller, movie.war,
             movie.western])

    movie_genre = np.array(movie_genre)
    cluster = KMeans(n_clusters=19)
    cluster.fit_predict(movie_genre)

    for i in range(0, n_users):
        average = np.zeros(19)
        tmp = []
        for m in range(0, 19):
            tmp.append([])
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
    utility_clustered2 = np.array(utility_clustered)

    # her kullanıcının verdiği oyların ortalamaları User objesinde tutuluyor.
    for i in range(0, n_users):
        x = utility_clustered2[i]
        user[i].avg_r = sum(a for a in x if a > 0) / sum(a > 0 for a in x)

    pcs_matrix = np.zeros((n_users, n_users))
    for i in range(0, n_users):
        for j in range(0, i):
            if i != j:
                pcs_matrix[i][j] = pearson(i + 1, j + 1)
                # sys.stdout.write("\rSimilarity Matrix [%d:%d] = %f" % (i + 1, j + 1, pcs_matrix[i][j]))
                # sys.stdout.flush()
                # time.sleep(0.00005)
    # print("\rSimilarity Matrix [%d:%d] = %f" % (i + 1, j + 1, pcs_matrix[i][j]))
    # print(pcs_matrix)
    # relate with ant colony pheromones
    graph = AntGraph(n_users, pcs_matrix)
    graph.reset_tau()
    num_iterations = 3
    # n_users = 5
    ant_colony = antcolony.AntColony(graph, 5, num_iterations)
    ant_colony.start()

    graph.delta_mat = isaverage(graph.delta_mat)
    result = connected_component_labelling(graph.delta_mat, 4)
    print(result)

    findresult(result)

# self transition da max node hatası ile ilgilenelim.
# result matirisi problemi:  bir kullanıcı birden fazla clusterda olmamalı, yalnızca birinde olmalı
# halihazırda result kullanıcı kullanıcı matrisi, ama  kullanıcı - cluster matrisine dönüştürmeliyiz.
# KMN kmeans neighboring fonksiyonuna benzer bir çalışma yapmalıyız.
if __name__ == '__main__':
    run()
