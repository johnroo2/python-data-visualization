from numpy import unique, where
from timeit import default_timer
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture

training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

start = default_timer()
gaussian_model = GaussianMixture(n_components=5)
gaussian_model.fit(training_data)
gaussian_result = gaussian_model.predict(training_data)
gaussian_clusters = unique(gaussian_result)

for gaussian_cluster in gaussian_clusters:
    index = where(gaussian_result == gaussian_cluster)
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

stop = default_timer()
print('Time: ', stop - start)  
pyplot.show()