#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, RocCurveDisplay, accuracy_score, confusion_matrix, ConfusionMatrixDisplay



water = pd.read_csv('~/Documents/metis/classification/project/water_potability.csv')





X = water[['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']]
y = water['Potability']



sns.pairplot(water,hue='Potability',kind='hist')


# lots of overlap on the various metrics means we won't get any one breakout winner



scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled)
X_scaled.columns = X.columns



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)




X_train_imputed = X_train.copy()
X_val_imputed = X_val.copy()
X_test_imputed = X_test.copy()
for field in X_train_imputed.columns:
    X_test_imputed[field] = X_test_imputed[field].fillna(X_train_imputed[field].mean())
    X_val_imputed[field] = X_val_imputed[field].fillna(X_train_imputed[field].mean())
    X_train_imputed[field] = X_train_imputed[field].fillna(X_train_imputed[field].mean())




X_train_invalid = X_train.copy()
X_val_invalid = X_val.copy()
X_test_invalid = X_test.copy()
for field in X_train_imputed.columns:
    X_test_invalid[field] = X_test_invalid[field].fillna(-999)
    X_val_invalid[field] = X_val_invalid[field].fillna(-999)
    X_train_invalid[field] = X_train_invalid[field].fillna(-999)


# ## Logistic Regression 
# ### Score 0.5812
# NaN converted to mean values




lr1 = LogisticRegression(C=1000, class_weight={0:1,1:3})
lr1.fit(X_train_imputed, y_train)

y_pred_lr1 = lr1.predict(X_val_imputed)

lr1.score(X_val_imputed,y_val)





print(recall_score(y_pred_lr1,y_val))
print(precision_score(y_pred_lr1,y_val))
print(f1_score(y_pred_lr1,y_val))


# high recall score is good because all the potable water was returned, but very low precision means we marked too many bodies as potable when they were not. This would be a bad model we can return to tune hyperparameters.

# ## K Nearest Neighbors
# ### Score 0.6117
# NaN converted to mean values




knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train_imputed, y_train)

y_pred_knn = knn.predict(X_val_imputed)

knn.score(X_val_imputed, y_val)





print(recall_score(y_pred_knn,y_val))
print(precision_score(y_pred_knn,y_val))
print(f1_score(y_pred_knn,y_val))


# Much better precision score in this model, not enough to use without major improvements

# ## Simple Decision Tree
# ### Score 0.5702
# NaN converted to -999




dtree = DecisionTreeClassifier(max_depth=7,min_samples_leaf=50,max_features=0.7,class_weight={0:1,1:3})
dtree.fit(X_train_invalid, y_train)

y_pred_tree = dtree.predict(X_val_invalid)

dtree.score(X_val_invalid,y_val)


print(recall_score(y_pred_tree,y_val))
print(precision_score(y_pred_tree,y_val))
print(f1_score(y_pred_tree,y_val))



# More improvement to precision score, using more advanced decision trees may provide an even better precision

# ## Extra Tree (Extremely Random)
# ### Score 0.5433
# NaN converted to -999




etree = ExtraTreeClassifier(max_depth=100, max_features=0.6, class_weight={0:1,1:7})
etree.fit(X_train_invalid, y_train)

y_pred_etree = etree.predict(X_val_invalid)

etree.score(X_val_invalid,y_val)

print(recall_score(y_pred_etree,y_val))
print(precision_score(y_pred_etree,y_val))
print(f1_score(y_pred_etree,y_val))

# The extremely random tree has a lower score, perhaps running the model with samples can produce a better result

# ## Random Forest
# ### Score 0.6507
# NaN converted to -999




rforest = RandomForestClassifier(n_estimators=150, max_depth=9, class_weight={0:2,1:3})
rforest.fit(X_train_invalid, y_train)

# y_pred_rforest = rforest.predict(X_val_invalid)
y_pred_rforest = rforest.predict_proba(X_val_invalid)[:,1]>=0.3

rforest.score(X_val_invalid,y_val)





print(recall_score(y_pred_rforest,y_val))
print(precision_score(y_pred_rforest,y_val))
print(f1_score(y_pred_rforest,y_val))


# Low precision score but a good recall means we could tune hyper parameters to favor precision without sacrificing too much

# ## Bagging Classifier
# ### Score 0.6324
# Nan converted to -999




bag = BaggingClassifier(n_estimators=50, max_samples=0.8)
bag.fit(X_train_invalid, y_train)

y_pred_bag = bag.predict(X_val_invalid)

bag.score(X_val_invalid,y_val)





print(recall_score(y_pred_bag,y_val))
print(precision_score(y_pred_bag,y_val))
print(f1_score(y_pred_bag,y_val))

# similar results to the random forest, this is another candidate for hyper parameter tuning

# ## Extra Trees Classifier
# ### Score 0.6166
# NaN converted to -999




etrees = ExtraTreesClassifier(n_estimators=500,max_depth=12)
etrees.fit(X_train_invalid, y_train)

y_pred_etrees = etrees.predict(X_val_invalid)

etrees.score(X_val_invalid,y_val)


print(recall_score(y_pred_etrees,y_val))
print(precision_score(y_pred_etrees,y_val))
print(f1_score(y_pred_etrees,y_val))


# Extra Trees (forest?) has a slightly better performance than the singular Extra Tree algorithm

# ## XGBoost
# ### Score 0.668
# NaN converted to -999


gbm = xgb.XGBClassifier( 
                        n_estimators=150,
                        max_depth=5,
                        objective='binary:logistic', #new objective
                        learning_rate=.02, 
                        subsample=.15,
                        min_child_weight=3,
                        colsample_bytree=0.8,
                        eval_metric='error',
                        early_stopping_rounds=50
                       )

eval_set=[(X_train_invalid,y_train),(X_val_invalid,y_val)]
fit_model = gbm.fit( 
                    X_train_invalid, y_train, 
                    eval_set=eval_set,
                    verbose=False
                   )

accuracy_score(y_val, gbm.predict(X_val_invalid, ntree_limit=gbm.best_ntree_limit)) 

# y_pred_xgboost = gbm.predict(X_val_invalid)

y_pred_xgboost = gbm.predict_proba(X_val_invalid, ntree_limit=gbm.best_ntree_limit)[:,1]>=0.4


print(recall_score(y_pred_xgboost,y_val))
print(precision_score(y_pred_xgboost,y_val))
print(f1_score(y_pred_xgboost,y_val))


models = pd.DataFrame(['Logistic Regression',
              'K Nearest Neighbors',
              'Decision Tree',
              'Extra Tree',
              'Random Forest',
              'Bagging',
              'Extra Trees',
              'XGBoost']
             )
model_dict2 = {'Logistic Regression':[lr1,X_val_imputed],
              'K Nearest Neighbors':[knn,X_val_imputed],
              'Decision Tree':[dtree,X_val_invalid],
              'Extra Tree':[etree,X_val_invalid],
              'Random Forest':[rforest,X_val_invalid],
              'Bagging':[bag,X_val_invalid],
              'Extra Trees':[etrees,X_val_invalid],
              'XGBoost':[gbm,X_val_invalid]
             }

# model_dict['XGBoost'][0].predict_proba(model_dict['XGBoost'][1],ntree_limit=gbm.best_ntree_limit)[:,1]>=0.4
# RocCurveDisplay.from_estimator(model_dict[1], model_dict[2], y_val)

st.write("""
# Model Water Source Predictions

This app helps identify which models show success and allows classification probability tuning
""")

model = st.sidebar.selectbox('Which model should be shown?',models)
# hyper_param = st.sidebar.checkbox('Custom Hyperparameters?', value=False)
# hyper_param2 = hyper_param == 0

# hyper = st.text_input('Optional: Change the hyper parameters', '', disabled=hyper_param2)

st.subheader(model)
st.write(model_dict2[model][0])
RocCurveDisplay.from_estimator(model_dict2[model][0], model_dict2[model][1], y_val)
plt.savefig('roc-curve.png')
st.image('roc-curve.png')

proba = st.sidebar.number_input('Select Probability Lower Limit for *Safe* Classification', min_value=0.0, max_value=1.0,
                value=0.5, step=0.05)

cm = confusion_matrix(y_val,model_dict2[model][0].predict_proba(model_dict2[model][1])[:,1]>=proba)

ConfusionMatrixDisplay(cm,display_labels=['unsafe','safe']).plot()
plt.savefig('cm-plot.png')
st.image('cm-plot.png')