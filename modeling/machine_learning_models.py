# Modelos baseados em ML como SVM, RF, ANN
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split

from EcoDistrib.common import msg_logger
from EcoDistrib.outputs import MapGenerator
from EcoDistrib.modeling import ModelDataPrepare

class MLModeling:
    def __init__(self):
        self.logger = msg_logger
        self.model_type = None

    def sdm_svm(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            presence_col='presence',
            kernel='rbf',
            C=1.0,
            gamma='scale',
            normalize=True,
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_svm.tif',
            pseudo_absence_ratio=0.3
        ):
        """
        Aplica o modelo SVM para predizer a distribuição das espécies.

        Parâmetros:
        - occurrence_data (pd.DataFrame): 
            Dados de ocorrência contendo latitude, longitude e presença/ausência.
        - tiff_paths (str ou list): 
            Caminhos para os arquivos TIFF com dados ambientais.
        - lat_col (str): 
            Nome da coluna com a latitude no DataFrame.
        - lon_col (str): 
            Nome da coluna com a longitude no DataFrame.
        - presence_col (str): 
            Nome da coluna com informações de presença (1) ou ausência (0).
        - kernel (str): 
            Tipo de kernel do SVM ('linear', 'poly', 'rbf', 'sigmoid').
        - C (float): 
            Parâmetro de regularização do SVM.
        - gamma (str ou float): 
            Coeficiente do kernel. Pode ser 'scale', 'auto' ou um valor float.
        - normalize (bool): 
            Se True, normaliza os dados de entrada.
        - save (bool): 
            Se True, salva o mapa resultante como um arquivo TIFF.
        - output_save (str): 
            Caminho do arquivo para salvar o mapa resultante.

        Logs:
        - Informações e erros são registrados usando `self.logger`.

        Retorna:
        - prediction_map (np.ndarray): 
            Mapa resultante com as previsões do modelo SVM.
        """
        self.model_type = 'SVM'
        try:
            # Verificar e criar a coluna de presença, se necessário
            if presence_col not in occurrence_data.columns:
                occurrence_data[presence_col] = 1
                self.logger.info(f"A coluna '{presence_col}' não foi encontrada. Criada com valores iguais a 1.")

            # Gerar pseudoausências, se necessário
            if not occurrence_data[presence_col].isin([0]).any():
                pseudo_ausencia_df = ModelDataPrepare().generate_pseudo_absence(
                    occurrence_data,
                    tiff_paths,
                    n_pseudo_ausencias=int(len(occurrence_data) * pseudo_absence_ratio),
                    presence_col=presence_col,
                    lat_col=lat_col,
                    lon_col=lon_col
                )
                pseudo_ausencia_df[presence_col] = 0
                occurrence_data = pd.concat([occurrence_data, pseudo_ausencia_df], ignore_index=True)
                self.logger.info("Pseudoausências geradas e adicionadas aos dados de ocorrência.")

            # Preparar os dados de raster
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)

            # Definir X e y
            X = raster_values
            y = occurrence_data[presence_col]

            # Dividir os dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.logger.info("Dados divididos em treino e teste.")

            # Normalizar os dados, se necessário
            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                self.logger.info("Os dados foram normalizados.")
            else:
                self.logger.info("Normalização desativada.")

            # Inicializar e treinar o modelo SVM
            model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
            model.fit(X_train, y_train)
            self.logger.info("Modelo SVM treinado com sucesso.")

            # Preparar a matriz de previsões para o mapa resultante
            X_pred = matriz.reshape(-1, matriz.shape[2])
            nan_rows = np.isnan(X_pred).any(axis=1)
            X_pred_valid = X_pred[~nan_rows]

            # Normalizar dados para previsões, se necessário
            if normalize:
                X_pred_valid = scaler.transform(X_pred_valid)

            # Previsão de probabilidades
            prediction_valid = model.predict_proba(X_pred_valid)[:, 1]

            # Criar um array de previsão completo e definir NaNs onde apropriado
            prediction_map = np.full(X_pred.shape[0], np.nan)
            prediction_map[~nan_rows] = prediction_valid
            prediction_map = prediction_map.reshape(matriz.shape[0], matriz.shape[1])

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(prediction_map, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em '{output_save}'.")

            return prediction_map

        except Exception as e:
            self.logger.error(f"Erro na execução da função `sdm_svm`: {e}")
            raise

    def optimize_rf_parameters(
            self,
            X,
            y,
            param_grid=None,
            cv=5,
            scoring='accuracy'
        ):
        """
        Realiza a otimização de hiperparâmetros para o modelo Random Forest.

        Parâmetros:
        - X (pd.DataFrame ou np.ndarray):
            Matriz de características com os dados ambientais (rasters).
        - y (pd.Series ou np.ndarray):
            Rótulos de presença/ausência para a espécie.
        - param_grid (dict, opcional):
            Dicionário com os parâmetros para ajustar no Random Forest.
            Padrão inclui 'n_estimators', 'max_depth', 'min_samples_split' e 'min_samples_leaf'.
        - cv (int, opcional):
            Número de divisões para validação cruzada (padrão: 5).
        - scoring (str, opcional):
            Métrica usada para avaliar a performance do modelo (padrão: 'accuracy').
                from sklearn.metrics import SCORERS

                # Listar todas as opções de scoring
                print(SCORERS.keys())
            Possibilidades para scoring:
            - For classification:
                - 'accuracy': Classification accuracy.
                - 'f1': F1 score (harmonic mean of precision and recall).
                - 'precision': Precision (positive predictive value).
                - 'recall': Recall (sensitivity).
                - 'roc_auc': Area under the ROC curve (binary classification).
                - 'average_precision': Average precision (for imbalanced datasets).

        Retornos:
        - best_model (RandomForestClassifier):
            Modelo Random Forest treinado com os melhores parâmetros.
        - best_params (dict):
            Dicionário dos melhores hiperparâmetros encontrados.
        - best_score (float):
            Pontuação validada cruzadamente do melhor modelo.

        Logs:
        - Informações e erros são registrados usando `self.logger`.
        """
        try:
            # Grid de parâmetros padrão, se nenhum for fornecido
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
                self.logger.info("Nenhum param_grid fornecido. Usando o grid padrão.")

            # Inicializar o modelo Random Forest
            rf = RandomForestClassifier(random_state=42)
            self.logger.info("Modelo Random Forest inicializado.")

            # Configurar o GridSearchCV para busca de melhores parâmetros
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=1
            )
            self.logger.info("GridSearchCV configurado com os parâmetros fornecidos.")

            # Ajustar a busca nos dados
            grid_search.fit(X, y)
            self.logger.info("GridSearchCV ajustado nos dados de treinamento.")

            # Extrair o melhor modelo, parâmetros e pontuação
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            self.logger.info(f"Melhores parâmetros encontrados: {best_params}")
            self.logger.info(f"Melhor pontuação (CV): {best_score:.4f}")

            return best_model, best_params, best_score

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante a otimização de parâmetros do Random Forest: {e}")
            raise

    def sdm_rf(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            presence_col='presence',
            optimize_params=True,
            param_grid=None,
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_rf.tif',
            pseudo_absence_ratio=0.3
        ):
        """
        Aplica o modelo Random Forest para predizer a distribuição das espécies.

        Parâmetros:
        - occurrence_data (pd.DataFrame):
            Dados de ocorrência contendo latitude, longitude e presença/ausência.
        - tiff_paths (str ou list):
            Caminhos para os arquivos TIFF com dados ambientais.
        - lat_col (str):
            Nome da coluna com a latitude no DataFrame.
        - lon_col (str):
            Nome da coluna com a longitude no DataFrame.
        - presence_col (str):
            Nome da coluna com informações de presença (1) ou ausência (0).
        - optimize_params (bool):
            Se True, realiza otimização de hiperparâmetros do Random Forest.
        - param_grid (dict, opcional):
            Dicionário com os parâmetros para otimização, se `optimize_params=True`.
        - save (bool, opcional):
            Se True, salva o mapa de predição como um arquivo TIFF.
        - output_save (str, opcional):
            Caminho do arquivo para salvar o mapa resultante.

        Logs:
        - Informações e erros são registrados usando `self.logger`.
        """
        self.model_type = 'RandomForest'
        try:
            # Verificar e criar coluna de presença, se necessário
            if presence_col not in occurrence_data.columns:
                occurrence_data[presence_col] = 1
                self.logger.info(f"A coluna '{presence_col}' não foi encontrada. Criada com valores iguais a 1.")

            # Gerar pseudoausências, se necessário
            if not occurrence_data[presence_col].isin([0]).any():
                pseudo_ausencia_df = ModelDataPrepare().generate_pseudo_absence(
                    occurrence_data,
                    tiff_paths,
                    n_pseudo_ausencias=int(len(occurrence_data) * pseudo_absence_ratio),
                    presence_col=presence_col,
                    lat_col=lat_col,
                    lon_col=lon_col
                )
                pseudo_ausencia_df[presence_col] = 0
                occurrence_data = pd.concat([occurrence_data, pseudo_ausencia_df], ignore_index=True)
                self.logger.info("Pseudoausências geradas e adicionadas aos dados de ocorrência.")

            # Preparar os dados de raster
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)

            # Definir X e y
            X = raster_values
            y = occurrence_data[presence_col]

            # Dividir os dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.logger.info("Dados divididos em treino e teste.")

            # Otimizar parâmetros ou treinar modelo padrão
            if optimize_params:
                self.logger.info("Iniciando otimização de parâmetros do Random Forest.")
                rf_model, best_params, best_score = self.optimize_rf_parameters(X_train, y_train, param_grid)
                self.logger.info(f"Parâmetros otimizados: {best_params}")
                self.logger.info(f"Melhor score (validação cruzada): {best_score:.4f}")
            else:
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, y_train)
                self.logger.info("Modelo Random Forest treinado com parâmetros padrão.")

            # Prever distribuição usando a matriz raster
            X_pred = matriz.reshape(-1, matriz.shape[2])
            nan_rows = np.isnan(X_pred).any(axis=1)
            X_pred_valid = X_pred[~nan_rows]

            # Predizer probabilidades para a classe de presença
            prediction_valid = rf_model.predict_proba(X_pred_valid)[:, 1]

            # Criar mapa de predições com NaNs nos locais apropriados
            prediction_map = np.full(X_pred.shape[0], np.nan)
            prediction_map[~nan_rows] = prediction_valid
            prediction_map = prediction_map.reshape(matriz.shape[0], matriz.shape[1])

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(prediction_map, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em '{output_save}'.")

            return prediction_map

        except Exception as e:
            self.logger.error(f"Erro na execução da função `sdm_rf`: {e}")
            raise

    def sdm_ann(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            presence_col='presence',
            hidden_layer_sizes=(50, 30),
            activation='relu',
            solver='adam',
            max_iter=500,
            normalize=True,
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_ann.tif',
            pseudo_absence_ratio=0.3
        ):
        """
        Aplica o modelo de Rede Neural Artificial (ANN) para predizer a distribuição das espécies.

        Parâmetros:
        - occurrence_data (pd.DataFrame): 
            Dados de ocorrência contendo latitude, longitude e presença/ausência.
        - tiff_paths (str ou list): 
            Caminhos para os arquivos TIFF com dados ambientais.
        - lat_col (str): 
            Nome da coluna com a latitude no DataFrame.
        - lon_col (str): 
            Nome da coluna com a longitude no DataFrame.
        - presence_col (str): 
            Nome da coluna com informações de presença (1) ou ausência (0).
        - hidden_layer_sizes (tuple): 
            Tamanho das camadas escondidas na ANN.
        - activation (str): 
            Função de ativação para as camadas escondidas.
        - solver (str): 
            Algoritmo para otimização do treinamento.
        - max_iter (int): 
            Número máximo de iterações para o treinamento.
        - normalize (bool): 
            Se True, normaliza os dados de entrada.
        - save (bool): 
            Se True, salva o mapa resultante como um arquivo TIFF.
        - output_save (str): 
            Caminho do arquivo para salvar o mapa resultante.

        Logs:
        - Informações e erros são registrados usando `self.logger`.
        """
        self.model_type = 'ANN'
        try:
            # Verificar e criar a coluna de presença, se necessário
            if presence_col not in occurrence_data.columns:
                occurrence_data[presence_col] = 1
                self.logger.info(f"A coluna '{presence_col}' não foi encontrada. Criada com valores iguais a 1.")

            # Gerar pseudoausências, se necessário
            if not occurrence_data[presence_col].isin([0]).any():
                pseudo_ausencia_df = ModelDataPrepare().generate_pseudo_absence(
                    occurrence_data,
                    tiff_paths,
                    n_pseudo_ausencias=int(len(occurrence_data) * pseudo_absence_ratio),
                    presence_col=presence_col,
                    lat_col=lat_col,
                    lon_col=lon_col
                )
                pseudo_ausencia_df[presence_col] = 0
                occurrence_data = pd.concat([occurrence_data, pseudo_ausencia_df], ignore_index=True)
                self.logger.info("Pseudoausências geradas e adicionadas aos dados de ocorrência.")

            # Preparar os dados de raster
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)

            # Definir X e y
            X = raster_values
            y = occurrence_data[presence_col]

            # Dividir os dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.logger.info("Dados divididos em treino e teste.")

            # Normalizar os dados, se necessário
            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                self.logger.info("Os dados foram normalizados.")
            else:
                self.logger.info("Normalização desativada.")

            # Inicializar e treinar o modelo ANN
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                max_iter=max_iter,
                random_state=42
            )
            model.fit(X_train, y_train)
            self.logger.info("Modelo ANN treinado com sucesso.")

            # Preparar a matriz de previsões para o mapa resultante
            X_pred = matriz.reshape(-1, matriz.shape[2])
            nan_rows = np.isnan(X_pred).any(axis=1)
            X_pred_valid = X_pred[~nan_rows]

            # Normalizar dados para previsões, se necessário
            if normalize:
                X_pred_valid = scaler.transform(X_pred_valid)

            # Previsão de probabilidades
            prediction_valid = model.predict_proba(X_pred_valid)[:, 1]

            # Criar um array de previsão completo e definir NaNs onde apropriado
            prediction_map = np.full(X_pred.shape[0], np.nan)
            prediction_map[~nan_rows] = prediction_valid
            prediction_map = prediction_map.reshape(matriz.shape[0], matriz.shape[1])

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(prediction_map, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em '{output_save}'.")

            return prediction_map

        except Exception as e:
            self.logger.error(f"Erro na execução da função `sdm_ann`: {e}")
            raise
