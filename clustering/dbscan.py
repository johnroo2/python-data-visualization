from numpy import unique, where
from timeit import default_timer
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

start = default_timer()
dbscan_model = DBSCAN(eps=0.25, min_samples=9)
dbscan_model.fit(training_data)
dbscan_result = dbscan_model.fit_predict(training_data)
dbscan_clusters = unique(dbscan_result)

for dbscan_cluster in dbscan_clusters:
    index = where(dbscan_result == dbscan_cluster)
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

stop = default_timer()
print('Time: ', stop - start)  
pyplot.show()