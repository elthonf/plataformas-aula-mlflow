# Setup

1. Criar ambiente com Python 3.9.13
2. Instalar requisitos de `requirements.txt`


# Execução da interface gráfica:

## Por arquivo apenas (pasta mlruns)
```commandline
mlflow ui
```

## Com instância de servidor (sqlite)

```commandline
mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root mlflowruns
```

# Serving de modelo
Para servir o modelo, é preciso logar o modelo com `log_model`.  
Somente então, 
escolha uma porta não usada
``` 
mlflow models serve -m <PATH-DO_MODELO> -p <PORTA> --env-manager <Gestor de ambiente>
```
Exemplo2:
```commandline
mlflow models serve -m temp/2/f8072fb81f664daf96ce55b6fe7a7b44/artifacts/modelo_mlf -p 5010  --env-manager local
mlflow models serve -m mlruns/2/f8072fb81f664daf96ce55b6fe7a7b44/artifacts/modelo_mlf -p 5010  --env-manager local
```
