from numpy import unique, where
from timeit import default_timer
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

start = default_timer()
agglomerative_model = AgglomerativeClustering(n_clusters=5)
agglomerative_result = agglomerative_model.fit_predict(training_data)
agglomerative_clusters = unique(agglomerative_result)

for agglomerative_cluster in agglomerative_clusters:
    index = where(agglomerative_result == agglomerative_cluster)
    pyplot.scatter(training_data[index, 0], training_data[index, 1])
    
stop = default_timer()
print('Time: ', stop - start)  
pyplot.show()