---
layout: post
title:  "Jupyter macros"
date:   2017-12-11 16:03:47 +0100
categories: other
tags: tutorial jupyter-notebook
---
Working in data science involves data exploration, plotting and model training and evaluating. These taks are performed using multiple libraries like Pandas, Sklearn, matplotlib ...

Since most of the work is done in a Jupyter notebooks, it is sometime annoying to keep importing the same libraries to work with. These jupyter macros will save you the time next time you create a new Jupyter notebook.

In this tutorial, we describe a way to invoke all the libraries needed for work using two lines instead of the 20+ lines to invoke all needed libraries. We will do that using a Jupyter Macro.

I like to split my imports in two categories: imports for regression problems and import for classification problems.

Start by openning a new notebook and importing the usual libraries used for a classification problem for example.

```python
# Handle table-like data and matrices
import numpy as np
import pandas as pd

from collections import Counter

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
```

    /home/maviator/anaconda2/lib/python2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)


To save these in a macro, we use the following format

```python
%macro -q <macro_name> <cell_number>
```

Since we executed the previous code in the first cell, this is the corresponding command

```python
%macro -q __importClassification 1
```

Now this command is saved to be used for this session. To be able to use when we restart the kernel, it needs to be stored.

```python
%store __importClassification
```

    Stored '__importClassification' (Macro)


Now you can restart the kernel. To load the libraries for classification, we first load the macro then execute it.

```python
%store -r __importClassification # to load the macro
```


```python
__importClassification # executing the macro
```

    /home/maviator/anaconda2/lib/python2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)


Let's create another macro for regression problems this time.

```python
# Handle table-like data and matrices
import numpy as np
import pandas as pd

from collections import Counter

# Modelling Algorithms
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
%matplotlib inline
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6
```


Start by creating the macro

```python
%macro -q __importRegression 3
```

Then storing it

```python
%store __importRegression
```

    Stored '__importRegression' (Macro)

To load the macro

```python
%store -r __importRegression
```

To execute the macro

```python
__importRegression
```

    /home/maviator/anaconda2/lib/python2.7/site-packages/sklearn/cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)


You can also see all the created macros using this command

```python
%store
```

    Stored variables and their in-db values:
    __importClassification             -> IPython.macro.Macro(u"# Handle table-like data and
    __importRegression                 -> IPython.macro.Macro(u"# Handle table-like data and



Now you got your import calls automated.
