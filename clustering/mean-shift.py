from numpy import unique, where
from timeit import default_timer
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import MeanShift

training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

start = default_timer()
mean_model = MeanShift()
mean_result = mean_model.fit_predict(training_data)
mean_clusters = unique(mean_result)

for mean_cluster in mean_clusters:
    index = where(mean_result == mean_cluster)
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

stop = default_timer()
print('Time: ', stop - start)  
pyplot.show()