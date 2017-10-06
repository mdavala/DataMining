import numpy as np
import pandas as pd
import hyperopt
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget

#initialize data
train_data = pd.read_csv('sample.csv')
test_data = pd.read_csv('sample_test.csv')

train_data.isnull().sum(axis=0)
train_data.fillna(-999, inplace=True)

X = train_data.drop('loan_status', axis=1)
y = train_data.loan_status

categorical_features_indices = np.where(X.dtypes != np.float)[0]


X_train, X_validation, y_train, y_validation = train_test_split(X, y, 
                                                    train_size=0.8, 
                                                    random_state=1234)

model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=42
)

model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_validation, y_validation),
    plot=True
)

cv_data = cv(
    model.get_params(),
    Pool(X, label=y, cat_features=categorical_features_indices),
)

print ('Best validation accuracy score: {:.2f}±{:.2f} on step {:.2f}'.format(np.max(cv_data["b'Accuracy'_test_avg"]),
    cv_data["b'Accuracy'_test_stddev"][np.argmax(cv_data["b'Accuracy'_test_avg"])],
    np.argmax(cv_data["b'Accuracy'_test_avg"])))

print ('Precise validation accuracy score: {}'.format(np.max(cv_data["b'Accuracy'_test_avg"])))







