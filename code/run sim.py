# %%
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
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
def create_missing_at_random_data(nrow, prop, seed):
    X, y = make_circles(n_samples = nrow,  noise=0.2, random_state=seed)

    randIndx = np.random.rand(y.shape[0])
    y[randIndx <= prop] = -1
    
    return X, y

def create_missing_class_one_data(nrow, prop, seed):
    X, y = make_circles(n_samples = nrow,  noise=0.2, random_state=seed)

    randIndx = np.random.rand(y.shape[0])

    y[ (randIndx <= prop) & (y == 1) ] = -1
    
    return X, y

# %%
def run_sim(props, create_data_functions):

    pieces = []
    seed = 0
    for prop in props:
        for create_data in create_data_functions:
            for b in np.arange(0, 10):

            # create data
                seed += 1
                X_train, y_train = create_data(50000, prop, seed)

                seed += 1
                X_test, y_test = create_data(50000, 0, seed)

                # train model using missing labels
                model_semi = SelfTrainingClassifier(base_estimator = KNeighborsClassifier(n_neighbors = 9, n_jobs=-1))
                model_semi.fit(X_train, y_train)

                # train model with only known labels
                model_known = KNeighborsClassifier(n_neighbors = 9, n_jobs=-1)
                X_train = X_train[y_train != -1, :]
                y_train = y_train[y_train != -1]
                model_known.fit(X_train, y_train)

                # performance on unseen data
                AUC_semi = roc_auc_score(y_test, model_semi.predict_proba(X_test)[:, 1])
                AUC_known = roc_auc_score(y_test, model_known.predict_proba(X_test)[:, 1])

                # summarize results into data frame.
                piece = {'missingPattern': [create_data.__name__], 'prop':[prop], 'b':[b], 'AUC_semi':[AUC_semi], 'AUC_known':[AUC_known]}
                piece = pd.DataFrame(piece)
                pieces.append(piece)

    result = pd.concat(pieces)
    result['prop'] = np.round(result['prop'], 2)

    return(result)
 

##############
# Run simulation
##############

# %%
result = run_sim(np.arange(.05, 1, .05), [create_missing_at_random_data, create_missing_class_one_data])
result.sort_values(['missingPattern', 'prop', 'b'])

# %%
result.to_csv(path_or_buf = 'data/result.csv', index=False)


# %%
