#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import PolynomialFeatures
# Hyperparameter tuning of model
from sklearn.model_selection import GridSearchCV
from scipy.stats import loguniform


# TASK 1: A

# Load Dataset
def load_dataset(filename):
    # Feature names.
    df_names = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore',
                'Cscore', 'Impulsive', 'SS', 'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke',
                'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer',
                'VSA']
    # Load dataset.
    df = pd.read_csv(filename, names=df_names)
    # Get the data type for each feature and more information for the data.
    print("DATASET INFO")
    print(df.info(verbose=True))
    print("\n MISSING VALUES COUNT")
    # Check if there are any missing value in the data.
    print(df.isnull().sum())

    return df

df = load_dataset("drug_consumption.data")

# Clean Data
y = df['Nicotine'] # make a copy of nicotine values.

def clean_data(df):
    # Replace values for countries and ethnicity
    df['Country'].replace(
        {-0.09765: 'Australia', 0.24923: 'Canada', -0.46841000000000005: 'NewZealand', -0.28519: 'Other',
         0.21128000000000002: 'Ireland', 0.9608200000000001: 'UK', -0.57009: 'USA'}, inplace=True)
    df['Ethnicity'].replace(
        {-0.50212: 'Asian', -1.1070200000000001: 'Black', 1.90725: 'Mixed-Black/Asian', 0.12600: 'Mixed-White/Asian',
         -0.22166: 'Mixed-White/Black', 0.11440: 'Other', -0.31685: 'White'}, inplace=True)
    df.drop(df.iloc[:, 13:32], inplace=True, axis=1)  # Drop every column from 13 to 32.
    df = df.drop('ID', axis=1)  # Drop ID column as its not relevant to the classification task.


clean_data(df) # Call method to clean data.


# TASK 1: B and D
# Split the data into 80% training, 20% testing.
np.random.seed(20)
# Split into train & test set
X_train, X_test, y_train, y_test = train_test_split(df,
                                                    y,
                                                    test_size=0.20, stratify=y)

# Confirm that the proportions in the dataset were maintained.
print(f"Proportions of y labels in Training Set: {Counter(y_train)}")
print(f"Proportions of y labels in Training Set: {Counter(y_test)}")


# TASK 1: C
# Encoding
def encode_data(data):
    # One-hot encode country and ethnicity (eliminate false distance between them).
    categories = ['Country', 'Ethnicity']
    data = pd.get_dummies(data, columns=categories)

    # Ordinal Encode Age, Gender, and Education as they follow a logical ordering.
    encoder = OrdinalEncoder()
    age_ordinal = data['Age'].values.reshape(-1, 1)
    data['Age'] = encoder.fit_transform(age_ordinal)

    gender_ordinal = data['Gender'].values.reshape(-1, 1)
    data['Gender'] = encoder.fit_transform(gender_ordinal)

    education_ordinal = data['Education'].values.reshape(-1, 1)
    data['Education'] = encoder.fit_transform(education_ordinal)

    return data

# Encode data X_train.
X_train = encode_data(X_train)

# Encode y.
encoder = OrdinalEncoder()
y_ordinal = y.values.reshape(-1, 1)
y = encoder.fit_transform(y_ordinal)


# TASK 2: A UNBALANCED
np.random.seed(20)
unbalanced_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='none', class_weight=None))
unbalanced_lr.fit(X_train, y_train)

scores = cross_val_score(unbalanced_lr, X_train, y_train, cv=5)
print("pipe_clf CV: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

unbalanced_train_preds = unbalanced_lr.predict(X_train)
f1_unbal = f1_score(y_train, unbalanced_train_preds, average=None)

# Hyperparameters tuning
print(unbalanced_lr.get_params().keys())

space = dict()
space['logisticregression__solver'] = ['newton-cg', 'lbfgs', 'liblinear', 'saga', 'sag']
space['logisticregression__penalty'] = ['none', 'l1', 'l2']
space['logisticregression__C'] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
space['logisticregression__tol'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
space['logisticregression__multi_class'] = ['ovr', 'multinomial']
space['logisticregression__max_iter'] = [1, 10, 20, 50, 100, 200, 300, 400, 600, 800, 1000]

# grid = dict( logisticregression__max_iter = max_iter, logisticregression__C = C, logisticregression__penalty = penalty, logisticregression__multi_class = multi_class)

grid_search = GridSearchCV(estimator=unbalanced_lr, param_grid=space, n_jobs=-1, cv=3, scoring='accuracy',
                           error_score=0)
grid_model = grid_search.fit(X_train, y_train)

print('Best Score: %s' % grid_model.best_score_)
print('Best Hyperparameters: %s' % grid_model.best_params_)

# TASK 2: C BALANCED

np.random.seed(20)
balanced_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='none', class_weight='balanced'))
balanced_lr.fit(X_train, y_train)

scores_b = cross_val_score(balanced_lr, X_train, y_train, cv=5)
print(scores_b)

print("pipe_clf CV: %0.2f accuracy with a standard deviation of %0.2f" % (scores_b.mean(), scores_b.std()))

balanced_train_preds = balanced_lr.predict(X_train)
f1_bal = f1_score(y_train, balanced_train_preds, average=None)


# Compare Unbalanced vs Balanced models F1 Score.
font = {'family': 'normal',
        'size': 16}

plt.rc('font', **font)

# Plot f1_scores for unbalanced and balanced models.
plt.figure(figsize=(10, 10))
plt.xlabel("Class")
plt.ylabel("F1-score")
plt.title("Unbalanced vs Balanced Models - F1 Score")
plt.plot([0, 1, 2, 3, 4, 5, 6], f1_unbal, label="Unbalanced")
plt.plot([0, 1, 2, 3, 4, 5, 6], f1_bal, label="Balanced")
plt.legend(loc="upper left")


# TASK 2: D Predict, decision_function, Predict_proba

# In[ ]:


def obtain_class_results(model):
    '''
    Examine the difference between the three function used for classification results.
    '''
    print(f"Predict() on first element of X_train: {model.predict(X_train[:1])}")
    print(f"Predict_proba() on first element of X_train:{model.predict_proba(X_train[:1])}")
    print(f"Decision_function() on first element of X_train: {model.decision_function(X_train[:1])}")


# Unbalanced

obtain_class_results(unbalanced_lr)

# Balanced
obtain_class_results(balanced_lr)

# # Evaluation
# remember to encode testing data!

# In[ ]:


from sklearn.model_selection import cross_val_predict

# In[ ]:


# Encode data X_train.
X_test = encode_data(X_test)

# In[ ]:


# iterating the test columns
for col in X_test.columns:
    print(col)

# In[ ]:


# iterating the train columns
for col in X_train.columns:
    print(col)

# The columns Country_NewZealand, Ethnicity_Mixed-Black/Asian were missing from the testing dataset because they did not have any values.

# In[ ]:


# Insert
X_test.insert(13, 'Country_NewZealand', 0)
X_test.insert(19, 'Ethnicity_Mixed-Black/Asian', 0)

# In[ ]:


# iterating the columns
for col in X_test.columns:
    print(col)

# In[ ]:


from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, precision_score


# In[ ]:


def plot_conf_mx(y_preds):
    fig, ax = plt.subplots(figsize=[12, 12])

    conf_mx = confusion_matrix(y_test, y_preds)

    ax.matshow(conf_mx, cmap='seismic')

    for (i, j), z in np.ndenumerate(conf_mx):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


# In[ ]:


def precision_recall_score(y_preds):
    print(classification_report(y_test, y_preds))
    print(f"precision - average None: {(precision_score(y_test, y_preds, average=None))}")
    print(f"precision - average Micro: {(precision_score(y_test, y_preds, average='micro'))}")
    print(f"precision - average Macro: {(precision_score(y_test, y_preds, average='macro'))}")

    print(f"recall - average None: {(recall_score(y_test, y_preds, average=None))}")
    print(f"recall - average Micro: {(recall_score(y_test, y_preds, average='micro'))}")
    print(f"recall - average Macro: {(recall_score(y_test, y_preds, average='macro'))}")


# In[ ]:


def evaluate_model(model):
    # Make predictions on test set.
    y_preds = model.predict(X_test)
    print("TESTING EVALUATION")
    print(f"Accuracy score: {accuracy_score(y_test, y_preds)}")
    print(f"Balanced accuracy score: {balanced_accuracy_score(y_test, y_preds)}")
    plot_conf_mx(y_preds)
    precision_recall_score(y_preds)


# ## Unbalanced

# In[ ]:


evaluate_model(unbalanced_lr)

# ## Balanced

# In[ ]:


evaluate_model(balanced_lr)

# # Extensions

# ## A

# In[ ]:


np.random.seed(20)
a_lr = make_pipeline(StandardScaler(), LogisticRegression(penalty='l2'))
a_lr.fit(X_train, y_train)

scores = cross_val_score(a_lr, X_train, y_train, cv=5)
print(scores)

# ## B

# In[ ]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

# In[ ]:


X_scaled.shape

# In[ ]:


poly = PolynomialFeatures(degree=2)
X_pol = poly.fit_transform(X_scaled)

# In[ ]:


X_pol.shape

# In[ ]:


X_pol.get_feature_names()

# EXTENSIONS: TASK C

# ### Unbalanced

# In[ ]:


np.random.seed(20)
poly_unbalanced_lr = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), unbalanced_lr)
poly_unbalanced_lr.fit(X_train, y_train)

scores = cross_val_score(poly_unbalanced_lr, X_train, y_train, cv=5)
print(scores)

# In[ ]:


# Lets make predictions on test set.
y_preds_unb_poly = poly_unbalanced_lr.predict(X_test)

# In[ ]:


precision_recall_score(y_preds_unb_poly)

# ### Balanced

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures

np.random.seed(20)
poly_balanced_lr = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), balanced_lr)
poly_balanced_lr.fit(X_train, y_train)

scores = cross_val_score(poly_balanced_lr, X_train, y_train, cv=5)
print(scores)

# In[ ]:


# Lets make predictions on test set.
y_preds_bal_poly = poly_balanced_lr.predict(X_test)



precision_recall_score(y_preds_bal_poly)

# EXTENSIONS: TASK D
print(classification_report(y_test, y_preds))






