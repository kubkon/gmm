gmm
=======

My implementation of the Gaussian mixture model (GMM).

Pull requests, comments, and suggestions are welcomed!

Installation
============
I'm assuming you are wise and you're using virtualenv and virtualenvwrapper. If not, go and [install them now](http://virtualenvwrapper.readthedocs.org/en/latest/).

In order to install the package, run in the terminal:

``` console
$ python setup.py sdist
```

Then:

``` console
$ pip install dist/GMM-0.1.0.tar.gz
```

And you're done!

Basic usage
===========

A typical usage would look as follows:

``` python
from pprint import pprint

import numpy as np

from gmm.algorithm import GMM


# Read in dataset from file
with open('faithful.txt', 'rt') as f:
    data = []
    for row in f:
        cols = row.strip('\r\n').split(' ')
        data.append(np.fromiter(map(lambda x: float(x), cols), np.float))
    data = np.array(data)

# Initialize GMM algorithm
n_components = 2
gmm_model = GMM(n_components)

# Fit GMM to the data
gmm_model.fit(data)

# Print the results
for mean, cov, prob in zip(gmm_model.means, gmm_model.covariances, gmm_model.mixing_probs):
    print("Fitted mean vector:")
    pprint(mean)

    print("Fitted covariance matrix:")
    pprint(cov)

    print("Fitted mixing probability:")
    pprint(prob)
```

More examples
=============
You can find more examples in the ```examples/``` folder.

License
=======

License information can be found in License.txt.