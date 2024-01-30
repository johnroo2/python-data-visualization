from numpy import unique, where
from timeit import default_timer
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import Birch

training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

start = default_timer()
birch_model = Birch(threshold=0.03, n_clusters=5)
birch_model.fit(training_data)
birch_result = birch_model.predict(training_data)
birch_clusters = unique(birch_result)

for birch_cluster in birch_clusters:
    index = where(birch_result == birch_cluster)
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

stop = default_timer()
print('Time: ', stop - start)  
pyplot.show()