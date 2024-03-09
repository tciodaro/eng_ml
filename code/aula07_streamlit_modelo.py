import streamlit as st
import pandas
import numpy as np
from sklearn import model_selection, tree, linear_model, metrics, feature_selection
import joblib

fname = '../data/dataset_vinhos.csv'
savefile = '../data/modelo_vinhos.pkl'

############################################ LEITURA DOS DADOS
print('=> Leitura dos dados')
df_wine = pandas.read_csv(fname,sep=';')
wine_target_col = 'target'
wine_label_map = df_wine[['target', 'target_label']].drop_duplicates()
drop_cols = ['target_label']
df_wine.drop(drop_cols, axis=1, inplace=True)
print(df_wine.head())

############################################ TREINO/TESTE E VALIDACAO
results = {}
for wine_type in df_wine['type'].unique():
    print('=> Training for wine:', wine_type)
    print('\tSeparacao treino/teste')
    wine = df_wine.loc[df_wine['type'] == wine_type].copy()
    Y = wine[wine_target_col]
    X = wine.drop([wine_target_col, 'type'], axis=1)
    ml_feature = list(X.columns)
    # train/test
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, Y, test_size=0.2)
    cvfold = model_selection.StratifiedKFold(n_splits = 10, random_state = 0, shuffle=True)
    print('\t\tTreino:', xtrain.shape[0])
    print('\t\tTeste :', xtest.shape[0])

    ############################################ GRID-SEARCH VALIDACAO CRUZADA
    print('\tTreinamento e hiperparametros')
    param_grid = {
        'C': [0.01, 0.1, 1],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    selector = feature_selection.RFE(tree.DecisionTreeClassifier(),
                                     n_features_to_select = 4)
    selector.fit(xtrain, ytrain)
    ml_feature = np.array(ml_feature)[selector.support_]
    
    model = model_selection.GridSearchCV(linear_model.LogisticRegression(),
                                         param_grid = param_grid,
                                         scoring = 'f1',
                                         refit = True,
                                         cv = cvfold,
                                         return_train_score=True
                                        )
    model.fit(xtrain[ml_feature], ytrain)

    ############################################ AVALIACAO GRUPO DE TESTE
    print('\tAvaliação do modelo')
    threshold = 0.5
    xtrain.loc[:, 'probabilidade'] = model.predict_proba(xtrain[ml_feature])[:,1]
    xtrain.loc[:, 'classificacao'] = (xtrain.loc[:, 'probabilidade'] > threshold).astype(int)
    xtrain.loc[:, 'categoria'] = 'treino'

    xtest.loc[:, 'probabilidade']  = model.predict_proba(xtest[ml_feature])[:,1]
    xtest.loc[:, 'classificacao'] = (xtest.loc[:, 'probabilidade'] > threshold).astype(int)
    xtest.loc[:, 'categoria'] = 'teste'

    wine = pandas.concat((xtrain, xtest))
    wine[wine_target_col] = pandas.concat((ytrain, ytest))
    wine['target_label'] = ['Alta Qualidade' if t else 'Baixa Qualidade'
                            for t in wine[wine_target_col]]
    
    print('\t\tAcurácia treino:', metrics.accuracy_score(ytrain, xtrain['classificacao']))
    print('\t\tAcurácia teste :', metrics.accuracy_score(ytest, xtest['classificacao']))

    ############################################ RETREINAMENTO DADOS COMPLETOS
    print('\tRetreinamento com histórico completo')
    model = model.best_estimator_
    model = model.fit(X[ml_feature], Y)
    
    ############################################ DADOS PARA EXPORTACAO
    results[wine_type] = {
        'model': model,
        'data': wine, 
        'features': ml_feature,
        'target_col': wine_target_col,
        'threshold': threshold
    }

############################################ EXPORTACAO RESULTADOS
print('=> Exportacao dos resultados')

joblib.dump(results, savefile, compress=9)
print('\tModelo salvo em', savefile)

