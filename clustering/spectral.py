from numpy import unique, where
from timeit import default_timer
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering

training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

start = default_timer()
spectral_model = SpectralClustering(n_clusters=5)
spectral_result = spectral_model.fit_predict(training_data)
spectral_clusters = unique(spectral_result)

for spectral_cluster in spectral_clusters:
    index = where(spectral_result == spectral_cluster)
    pyplot.scatter(training_data[index, 0], training_data[index, 1])
    
stop = default_timer()
print('Time: ', stop - start)  
pyplot.show()