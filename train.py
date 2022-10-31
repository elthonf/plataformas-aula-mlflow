import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import GradientBoostingRegressor as gbr
import joblib
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import uuid
import random
import json

import util

experiment_name = 'Risco de Credito V2'


# JOB de treinamento básico, utilizando o MLFLOW
if __name__ == "__main__":
    # Carrega os dados
    mydf = pd.read_csv('datasets/BaseDefault01.csv')

    # Identifica no dataset as variáveis independentes e a variavel alvo
    targetcol = 'default'
    y = mydf[targetcol]

    independentcols = ['renda', 'idade', 'etnia', 'sexo', 'casapropria', 'outrasrendas', 'estadocivil', 'escolaridade']
    X = mydf[independentcols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Obtém o experimento (cria se necessário
    experimento = mlflow.get_experiment_by_name(experiment_name)
    if experimento is None:
        mlflow.create_experiment(experiment_name)
        experimento = mlflow.get_experiment_by_name(experiment_name)


    # Realiza n treinamentos

    for i in range(20):
        runName = f"Run .. {util.get_a_funnyName()}"
        print(f"Experiment: {runName}")

        try:
            nruns_atual = len(mlflow.list_run_infos(experiment_id=experimento.experiment_id))
            run = mlflow.start_run(experiment_id=experimento.experiment_id, run_name=runName)

            if random.randint(1, 3) == 1:
                raise Exception("Chance de um erro aleatório a cada 1.")

            clf = rfc(n_estimators=random.randint(10, 50),
                      min_samples_leaf=random.randint(1, 5),
                      max_depth=None if random.randint(1, 2) == 1 else random.randint(10, 100)
                      )
            clf.fit(X=X_train[independentcols], y=y_train)
            clf.independentcols = independentcols

            clf_acuracia = clf.score(X=X_test[independentcols], y=y_test)
            print(f"Modelo (classificador), criado com acurácia de: [{clf_acuracia}]")

            #Log (sempre no run atual)
            mlflow.log_param("criterion", clf.criterion)
            mlflow.log_param("n_estimators", clf.n_estimators)
            mlflow.log_param("min_samples_leaf", clf.min_samples_leaf)
            mlflow.log_param("max_depth", clf.max_depth)
            mlflow.log_param("N.Run", nruns_atual + 1)
            mlflow.log_metric("acuracia", clf_acuracia)
            mlflow.log_metric("minha métrica customizada", 10)

            #Export transparente, gerenciado pelo mlflow
            mlflow.sklearn.log_model(clf, "modelo_mlf")

            #Export manual
            model_name = "nome_arquivo.pkl"
            filename = "modelo/" + model_name
            joblib.dump(value=clf, filename=filename)


            #Registro da pasta "modelo"
            mlflow.log_artifacts("./modelo", "modelo")
            mlflow.end_run()
            pass
        except Exception as err:
            print("Experimento com erro: " + str(err))

            mlflow.end_run("FAILED")
            pass


