# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:06:01 2016

@author: iceberg
"""

import gc
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from scipy.sparse import csr_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    def probToClass(prob):
        n = prob.shape[0]
        return [prob[i,].argmax() for i in range(n)]
    
    print('Started!')
    train = pd.read_csv('./input/train.csv', na_values='?')
    test = pd.read_csv('./input/test.csv', na_values='?')
    
    # remove replicated columns
    remove = []
    c = train.columns
    for i in range(len(c)-1):
        v = train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])
                remove.append(c[i])
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    # add Number of 22 meds not prescribed
    features = train.columns[24:46]
    train.insert(2, 'noMed', (train[features] != 'No').astype(int).sum(axis=1))
    test.insert(2, 'noMed', (test[features] != 'No').astype(int).sum(axis=1))
    
    #%% Encode categorical variables
    def preprocess(train):
        def catToNum(df, col, method):
            mapValue = df[col].value_counts()
            value = mapValue.index.tolist()
            if method == 'freq':
                encode = mapValue.values.tolist()
            if method == 'range':
                encode = [int(s.lstrip('[').lstrip('>').split('-')[0]) for s in value]
            if method == 'testresult':
                small_dic = {'None':0, 'Norm':1}
                encode = [s.lstrip('[').lstrip('>') for s in value]
                encode = [int(s) if s.isdigit() else s for s in encode]
                encode = [small_dic.get(s) if s in small_dic else s for s in encode]
            dic = dict(zip(value, encode))
            df[col] = df[col].map(lambda x: dic.get(x,-1))  # give 0 for nan
                    
        catToNum(train, 'race', 'freq')   
        catToNum(train, 'gender', 'freq')    
        catToNum(train, 'age', 'range') 
        catToNum(train, 'weight', 'range')
        train['admission_type_id'].value_counts(dropna=False)    
        train['discharge_disposition_id'].value_counts(dropna=False)    
        train['admission_source_id'].value_counts(dropna=False)    
        catToNum(train, 'payer_code', 'freq')
        catToNum(train, 'medical_specialty', 'freq')
        catToNum(train, 'diag_1', 'freq')
        catToNum(train, 'diag_2', 'freq')
        catToNum(train, 'diag_3', 'freq')
        
        catToNum(train, 'max_glu_serum', 'testresult')
        catToNum(train, 'A1Cresult', 'testresult')
        
        # 24 features
        dic = {'No':0, 'Down':1, 'Steady':2, 'Up':3}
        
        train['metformin'] = train['metformin'].map(lambda x: dic.get(x,-1))
        train['repaglinide'] = train['repaglinide'].map(lambda x: dic.get(x,-1))
        train['nateglinide'] = train['nateglinide'].map(lambda x: dic.get(x,-1))
        train['chlorpropamide'] = train['chlorpropamide'].map(lambda x: dic.get(x,-1))
        train['glimepiride'] = train['glimepiride'].map(lambda x: dic.get(x,-1))
        train['acetohexamide'] = train['acetohexamide'].map(lambda x: dic.get(x,-1))
        
        train['glipizide'] = train['glipizide'].map(lambda x: dic.get(x,-1))
        train['glyburide'] = train['glyburide'].map(lambda x: dic.get(x,-1))
        train['tolbutamide'] = train['tolbutamide'].map(lambda x: dic.get(x,-1))
        train['pioglitazone'] = train['pioglitazone'].map(lambda x: dic.get(x,-1))
        train['rosiglitazone'] = train['rosiglitazone'].map(lambda x: dic.get(x,-1))
        train['acarbose'] = train['acarbose'].map(lambda x: dic.get(x,-1))
        
        train['miglitol'] = train['miglitol'].map(lambda x: dic.get(x,-1))
        train['troglitazone'] = train['troglitazone'].map(lambda x: dic.get(x,-1))
        train['tolazamide'] = train['tolazamide'].map(lambda x: dic.get(x,-1))
        train['examide'] = train['examide'].map(lambda x: dic.get(x,-1))    
        train['insulin'] = train['insulin'].map(lambda x: dic.get(x,-1))
        train['glyburide-metformin'] = train['glyburide-metformin'].map(lambda x: dic.get(x,-1))
        
        train['glipizide-metformin'] = train['glipizide-metformin'].map(lambda x: dic.get(x,-1))    
        train['glimepiride-pioglitazone'] = train['glimepiride-pioglitazone'].map(lambda x: dic.get(x,-1))
        train['metformin-rosiglitazone'] = train['metformin-rosiglitazone'].map(lambda x: dic.get(x,-1))
        train['metformin-pioglitazone'] = train['metformin-pioglitazone'].map(lambda x: dic.get(x,-1))
        
        dic = {'No':0, 'Ch':1}
        train['change'] = train['change'].map(lambda x: dic.get(x,-1))
        dic = {'No':0, 'Yes':1}
        train['diabetesMed'] = train['diabetesMed'].map(lambda x: dic.get(x,-1))
    preprocess(train)
    preprocess(test)
    
    # add PCA features
    features = train.columns[2:-1]
    pca = PCA(n_components=2)

    pca.fit_transform(normalize(pd.concat([train[features], test[features]]), axis=0))
    x_train_projected = pca.transform(normalize(train[features], axis=0))
    x_test_projected = pca.transform(normalize(test[features], axis=0))  
    
    train['PCAOne'] = x_train_projected[:, 0]
    train['PCATwo'] = x_train_projected[:, 1]
    test['PCAOne'] = x_test_projected[:, 0]
    test['PCATwo'] = x_test_projected[:, 1]
        
    # re-admitted encoding
    dic = {'NO':0, '>30':1, '<30': 2}
    train['readmitted'] = train['readmitted'].map(lambda x: dic.get(x,-1))

    #%% feature importance 
    def misclassError(act, pred):
        return np.sum(act != pred) / len(act)
        
    features = train.columns[2:].tolist()
    features.remove('readmitted')
    y = train['readmitted']
    
    params = {'objective':'multi:softmax', 'num_class':3, 'eval_metric':'merror', 
              'max_depth': 3, 'min_child_weight':1, 'eta': 0.1, 'max_delta_step':1, 
              'seed': 12, 'silent': 1}
    num_rounds = 500

    dtrain = xgb.DMatrix(train[features], label=y)
    clf_ini = xgb.train(params, dtrain, num_rounds)
    first_result = clf_ini.predict(dtrain)    
    print('Training misclassfication error:', misclassError(y, first_result))
    
    outfile = open('xgb.fmap', 'w')    
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()
    importance = clf_ini.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    # Plotitup
    plt.figure()
    df.plot()
    df[1:20].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(5, 15))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('Feature_Importance_xgb.png')
    
    tokeep = df.ix[:,0].tolist()
    
    #%% Try a few algo: logistic regression, random forest, gradient boosting tree.
    
    y = train['readmitted']
    features = train.columns[2:].tolist()
    features.remove('readmitted') 
    X = train[tokeep]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    # logistic regression
    lg = LogisticRegression(penalty='l1', class_weight = 'balanced', 
                            random_state = 12)    
    lg = lg.fit(X_train, y_train)
    y_pred = lg.predict(X_test)
    print('Logistic regression error: ', misclassError(y_test, y_pred))
    print('Logistic regression f score: ', f1_score(y_test, y_pred))
    confusion_matrix(y_test, y_pred)
    
    # random forest
    rf = RandomForestClassifier(n_estimators=50,
                                n_jobs = 2, random_state = 12, 
                                class_weight = 'balanced')
    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print('Random forest error: ', misclassError(y_test, y_pred))
    print('Random forest f score: ', f1_score(y_test, y_pred))
    
    # xgboost
    xg = xgb.XGBClassifier()
    xg = xg.fit(X_train, y_train)
    y_pred = xg.predict(X_test)
    print('Xgboost error: ', misclassError(y_test, y_pred))
    print('Xgboost f score: ', f1_score(y_test, y_pred))
    confusion_matrix(y_test, y_pred)
    
#%% Feature engineering and exploration
            
    train['num_lab_procedures'].value_counts()
    train['num_lab_procedures'].hist(bins=500)
    
    var = 'change'
    sns.FacetGrid(train, hue="readmitted", size=10) \
       .map(plt.scatter, "PCAOne", var) \
       .add_legend()
    sns.FacetGrid(train[train[var]>=0], hue="readmitted", size=6) \
       .map(plt.hist, var, bins=10) \
       .add_legend()
    sns.FacetGrid(train[train[var]>=0], hue="readmitted", size=6) \
       .map(sns.kdeplot, var) \
       .add_legend()
    
    #train['log_number_emergency'] = train.number_emergency.map(lambda x: np.log(x+1))
    train['fe_number_outpatient'] = [1.0 if x < 1.0 else 0.0 for x in train['number_outpatient']]
    test['fe_number_outpatient'] = [1.0 if x < 1.0 else 0.0 for x in test['number_outpatient']]
    
    tokeep = tokeep + ['fe_number_outpatient']
# 'num_medications',
# 'num_lab_procedures',
# 'PCATwo',
# 'diag_3',
# 'age',***
# 'number_inpatient',***
# 'admission_source_id',
# 'PCAOne'
# 'payer_code',**3 dominent payer_code
# 'medical_specialty',
# 'time_in_hospital',
# 'number_outpatient',*** outlier AND if take < 5, a lot of 0-read has 0 outpatient visits, 
# 'admission_type_id',*** type 6 has more patients get read after 30
# 'num_procedures',
# 'number_diagnoses',* the % of 2 increases a lot for number_diagnoses = 6-9
# 'A1Cresult',
# 'insulin',
# 'number_emergency',* more patients get 1 and 2 if > 0
# 'weight',
# 'race',
# 'noMed',* get high chance to read if >= 6
# 'glipizide',
# 'metformin',
# 'gender',*  
# 'glimepiride',
# 'glyburide',
# 'rosiglitazone',
# 'pioglitazone',
# 'max_glu_serum',
# 'repaglinide',
# 'glyburide-metformin',
# 'nateglinide',
# 'acarbose',
# 'tolazamide',
# 'change',**higher chance to get admitted if ch
# 'miglitol',
# 'chlorpropamide',
# 'tolbutamide']
 

#%% Gridsearch for xgboost
    # search for optimal n_estimators with eta=0.1   
    def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        params = alg.get_xgb_params()
        params['num_class'] = 3
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['readmitted'].values)
        cvresult = xgb.cv(params, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['merror'], early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
        alg.fit(dtrain[predictors], dtrain['readmitted'],eval_metric='merror')
        
        dtrain_predictions = alg.predict(dtrain[predictors])
        print("\nModel Report")
        print("Misclassification error: ", misclassError(dtrain['readmitted'].values, dtrain_predictions))
        
    predictors = tokeep
    xgb1 = XGBClassifier(learning_rate =0.1,n_estimators=500,max_depth=5,
        min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,
        objective= 'multi:softmax',nthread=4,scale_pos_weight=1,seed=12)
    modelfit(xgb1, train, predictors)
    #result: 264
    
    # search for optimal max_depth, min_child_weight
    param_test1 = {
        'max_depth':[4,5,6],
        'min_child_weight':[1,3,5],
        #'eval_metric':['merror']
    }
    from sklearn.grid_search import GridSearchCV
    gsearch1 = GridSearchCV(estimator = \
        XGBClassifier(learning_rate =0.1, n_estimators=300, gamma=0, 
                      subsample=0.8, colsample_bytree=0.8,
                      objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=12), 
        param_grid = param_test1, cv=5)
    gsearch1.fit(train[predictors],train['readmitted'])
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    # result: {'max_depth': 4, 'min_child_weight': 3}, accuracy rate: 0.58037414172183799
    
    param_test2 = {
        'gamma':[0, 0.2, 0.4]
    }
    gsearch2 = GridSearchCV(estimator = \
        XGBClassifier(learning_rate =0.1, n_estimators=300, max_depth=4, min_child_weight=3, gamma=0, 
                      subsample=0.8, colsample_bytree=0.8,
                      objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=12), 
        param_grid = param_test2, cv=5)
    gsearch2.fit(train[predictors],train['readmitted'])
    gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
    # result: {'gamma': 0}, 0.58037414172183799
    
    param_test3 = {
        'subsample':[0.6, 0.8, 1.0],
        'colsample_bytree':[0.5, 0.7, 0.9]
    }
    gsearch3 = GridSearchCV(estimator = \
        XGBClassifier(learning_rate =0.1, n_estimators=300, max_depth=4, min_child_weight=3, gamma=0,
                      objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=12), 
        param_grid = param_test3, cv=5)
    gsearch3.fit(train[predictors],train['readmitted'])
    gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
    #{'colsample_bytree': 0.5, 'subsample': 1.0}, 0.58190952304914445
    
#%% Final model and output 
        
    features = train.columns[2:].tolist()
    features.remove('readmitted')
    todrop = list(set(features).difference(set(tokeep)))
    train.drop(todrop, inplace=True, axis=1)
    test.drop(todrop, inplace=True, axis=1)
    
    features = train.columns[2:].tolist()
    features.remove('readmitted') 
    split = 5
    skf = StratifiedKFold(train.readmitted.values,
                          n_folds=split,
                          shuffle=False,
                          random_state=None)

    num_rounds = 500  
    params = {'objective':'multi:softprob', 'num_class':3, 'eval_metric':'merror', 
              'max_depth':4, 'min_child_weight':2, 'eta': 0.1,
              'subsample':1, 'colsample_bytree':0.5, 'scale_pos_weight':1,
              'gamma':0, 'seed': 12, 'silent': 1}
    
    train_preds = None
    test_preds = None
    visibletrain = blindtrain = train
    index = 0
    for train_index, test_index in skf:
        print('Fold:', index)
        visibletrain = train.iloc[train_index]
        blindtrain = train.iloc[test_index]
        # imbalanced data        
        
        dvisibletrain = xgb.DMatrix(csr_matrix(visibletrain[features]),
                        visibletrain.readmitted.values,
                        silent=True)
        dblindtrain = xgb.DMatrix(csr_matrix(blindtrain[features]),
                        blindtrain.readmitted.values,
                        silent=True)
        watchlist = [(dblindtrain, 'eval'), (dvisibletrain, 'train')]
        clf = xgb.train(params, dvisibletrain, num_rounds,
                        evals=watchlist, early_stopping_rounds=50,
                        verbose_eval=False)

        blind_preds = clf.predict(dblindtrain)
        print('Misclassfication error:', \
            misclassError(blindtrain.readmitted.values, probToClass(blind_preds)))        
        index = index+1
        del visibletrain, blindtrain, dvisibletrain, dblindtrain
        gc.collect()
        
        dfulltrain = xgb.DMatrix(csr_matrix(train[features]),
                        train.readmitted.values,
                        silent=True)
        dfulltest = xgb.DMatrix(csr_matrix(test[features]),
                        silent=True)
        if(train_preds is None):
            train_preds = clf.predict(dfulltrain)
            test_preds = clf.predict(dfulltest)
        else:
            train_preds += clf.predict(dfulltrain)
            test_preds += clf.predict(dfulltest)
        del dfulltrain, dfulltest, clf
        gc.collect()
        
    
#%% Write output
    train_preds = train_preds/index
    test_preds = test_preds/index
    train_preds_class = probToClass(train_preds)
    test_preds_class = probToClass(test_preds)
    # Examine train result
    print('Average Misclassification Error:', misclassError(train.readmitted.values, train_preds_class))
    confusion_matrix(train.readmitted.values, train_preds_class)
    submission = pd.DataFrame({"readmitted": train.readmitted,
                               "PREDICTION": train_preds_class})
    submission.to_csv("Xgbtrain.csv", index=False)
    # Output test prediction
    test_output = ['NO' if o==0 else o for o in test_preds_class]
    test_output = ['>30' if o==1 else o for o in test_output]
    test_output = ['<30' if o==2 else o for o in test_output]
    submission = pd.DataFrame({"readmitted": test_output})
    submission.to_csv("Minh.csv", index=False)
    print('Finish')
