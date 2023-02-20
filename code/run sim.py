# %%
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_hastie_10_2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading
from sklearn.metrics import roc_auc_score

##############
# Set working directory
##############
# %%
os.chdir('S:/Python/projects/semi_supervised_two')

##############
# Define helpers
##############
# %%
def create_data(nrow, prop, seed):
    X, y = make_hastie_10_2(n_samples = nrow,  random_state=seed)
    y[y == -1] = 0

    randIndx = np.random.rand(y.shape[0])
    y[randIndx <= prop] = -1
    
    return X, y

##############
# Run simulation
##############

# %%
np.random.seed(42)
pieces = []
seed = 0
for prop in np.arange(.05, 1, .05):
    print(prop)
    for b in np.arange(0, 5):

        # create data
        seed += 1
        X_train, y_train = create_data(50000, prop, seed)

        seed += 1
        X_test, y_test = create_data(50000, 0, seed)

        # train model using missing labels
        model_semi = SelfTrainingClassifier(base_estimator = HistGradientBoostingClassifier())
        model_semi.fit(X_train, y_train)

        # train model with only known labels
        model_known = HistGradientBoostingClassifier()
        X_train = X_train[y_train != -1, :]
        y_train = y_train[y_train != -1]
        model_known.fit(X_train, y_train)

        AUC_semi = roc_auc_score(y_test, model_semi.predict_proba(X_test)[:, 1])
        AUC_known = roc_auc_score(y_test, model_known.predict_proba(X_test)[:, 1])

        piece = {'prop':[prop], 'b':[b], 'AUC_semi':[AUC_semi], 'AUC_known':[AUC_known]}
        piece = pd.DataFrame(piece)
        pieces.append(piece)

result = pd.concat(pieces)
result.head()

# %%
result['prop'] = np.round(result['prop'], 2)

# %%
result.to_csv(path_or_buf = 'data/result.csv', index=False)

# %%
