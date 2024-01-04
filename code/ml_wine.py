import pycaret.classification as pc
import pandas as pd
from sklearn import model_selection
import mlflow
import sys


# Argumentos para a rotina de treinamento
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
experiment_name = sys.argv[2] if len(sys.argv) > 2 else 'wine-ml-model'


## CONFIGURACAO
test_size = 0.2
model_name = 'classificacao_vinhos' # evitar espacos, -, e outros caracteres.
wine_target_col = 'target'
ignore_features = ['target_label']
categorical_features = ['type']


## LEITURA DOS DADOS DE TREINAMENTO
df_wine = pd.read_csv('../data/dataset_vinhos.csv',sep=';')
wine_label_map = df_wine[['target', 'target_label']].drop_duplicates()
print(df_wine.shape)
df_wine.head()

## TREINAMENTO DO MODELO
Y = df_wine[wine_target_col]
X = df_wine.drop(wine_target_col, axis=1)
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X, Y, test_size=test_size, random_state = seed)
df_train = xtrain.copy()
df_test = xtest.copy()
df_train[wine_target_col] = ytrain
df_test[wine_target_col] = ytest

# mlflow.set_tracking_uri("sqlite:///mlruns.db")

s = pc.setup(data = df_train, 
             target = wine_target_col,
             test_data=df_test,
             categorical_features = categorical_features,
             ignore_features = ignore_features,
             experiment_name = experiment_name, 
             log_experiment = True, 
             log_plots = True)
models = ['lr', 'dt', 'rf']
bestmodel = pc.compare_models(include = models)

classification_plots = [ 'auc', 
                        'confusion_matrix',
                        # 'error','class_report',
                         # 'learning','vc','feature',
                       ]
for plot_type in classification_plots:
    print('=> Aplicando plot ', plot_type)
    try:
        artifact = pc.plot_model(bestmodel, plot=plot_type, save=True)
        mlflow.log_artifact(artifact)
    except:
        print('=> Nao possivel plotar: ', plot_type )
        continue

mlflow.end_run()
        
