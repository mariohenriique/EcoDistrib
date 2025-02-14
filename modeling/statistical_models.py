# Funções baseadas em GLM, GAM, etc.
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pygam import GAM,terms,s

from EcoDistrib.common import msg_logger
from EcoDistrib.outputs import MapGenerator
from EcoDistrib.modeling import ModelDataPrepare

class StatisticalModeling:
    def __init__(self):
        self.logger = msg_logger
        self.model_type = None

    def sdm_gam(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            presence_col='presence',  # Coluna indicando presença (1) ou ausência (0)
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_gam.tif',
            pseudo_absence_ratio=0.3
        ):
        """
        Aplica o modelo GAM para predizer a distribuição das espécies.

        Parâmetros:
        - occurrence_data (pd.DataFrame):
            Dados de ocorrência, contendo as colunas de latitude, longitude e presença/ausência.
        - tiff_paths (str ou list):
            Caminho para os arquivos TIFF com dados ambientais.
        - lat_col (str, opcional):
            Nome da coluna com a latitude no DataFrame.
        - lon_col (str, opcional):
            Nome da coluna com a longitude no DataFrame.
        - presence_col (str, opcional):
            Nome da coluna com a informação de presença (1) ou ausência (0).
        - save (bool, opcional):
            Se True, salva o mapa resultante como um arquivo TIFF.
        - output_save (str, opcional):
            Caminho do arquivo para salvar o mapa resultante.

        Retorno:
        - np.ndarray:
            Array 2D com as probabilidades previstas para cada ponto do raster.

        Logs:
        - Mensagens de progresso e erros são registrados usando `self.logger`.
        """
        self.model_type = 'GAM'
        try:
            # Verificar se a coluna de presença está presente no DataFrame
            if presence_col not in occurrence_data.columns:
                occurrence_data[presence_col] = 1
                self.logger.warning(
                    f"A coluna '{presence_col}' não foi encontrada. Ela foi criada com valores iguais a 1."
                )

            # Gerar dados de pseudo ausência
            pseudo_ausencia_df = ModelDataPrepare().generate_pseudo_absence(
                occurrence_data,
                tiff_paths,
                n_pseudo_ausencias=int(len(occurrence_data) * pseudo_absence_ratio),
                presence_col=presence_col,
                lat_col=lat_col,
                lon_col=lon_col
            )

            pseudo_ausencia_df[presence_col] = 0  # Marcar pseudo-ausências como 0
            occurrence_data = pd.concat([occurrence_data, pseudo_ausencia_df], ignore_index=True)

            # Preparar os dados de raster
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)

            # Definir X (dados ambientais) e y (presença/ausência)
            X = raster_values
            y = occurrence_data[presence_col]

            # Ajustar o modelo GAM
            n_features = X.shape[1]
            termos = terms.TermList(*(s(i) for i in range(n_features)))  # Criar TermList explicitamente
            modelo = GAM(termos).fit(X, y)
            self.logger.info("Modelo GAM ajustado com sucesso.")

            # Prever utilizando a matriz 3D do raster
            X_pred = matriz.reshape(-1, matriz.shape[2])
            nan_rows = np.isnan(X_pred).any(axis=1)  # Identificar linhas com NaN
            X_pred_valid = X_pred[~nan_rows]  # Remover linhas com NaN
            previsao_valida = modelo.predict(X_pred_valid)

            # Reconstruir o array de previsão completo
            previsao_gam = np.full(X_pred.shape[0], np.nan)
            previsao_gam[~nan_rows] = previsao_valida
            previsao_gam = previsao_gam.reshape(matriz.shape[0], matriz.shape[1])

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(previsao_gam, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return previsao_gam

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante a aplicação do GAM: {e}")
            raise

    def sdm_glm(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            presence_col='presence',
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_glm.tif',
            pseudo_absence_ratio=0.3
        ):
        """
        Aplica o modelo GLM para predizer a distribuição das espécies.

        Parâmetros:
        - occurrence_data (pd.DataFrame):
            Dados de ocorrência, contendo as colunas de latitude, longitude e presença/ausência.
        - tiff_paths (str ou list):
            Caminho para os arquivos TIFF com dados ambientais.
        - lat_col (str, opcional):
            Nome da coluna com a latitude no DataFrame.
        - lon_col (str, opcional):
            Nome da coluna com a longitude no DataFrame.
        - presence_col (str, opcional):
            Nome da coluna com a informação de presença (1) ou pseudoausência (0).
        - save (bool, opcional):
            Se True, salva o mapa resultante como um arquivo TIFF.
        - output_save (str, opcional):
            Caminho do arquivo para salvar o mapa resultante.

        Retorno:
        - np.ndarray:
            Array 2D com as probabilidades previstas para cada ponto do raster.

        Logs:
        - Mensagens de progresso e erros são registrados usando `self.logger`.
        """
        self.model_type = 'GLM'
        try:
            # Verificar se a coluna de presença existe no DataFrame
            if presence_col not in occurrence_data.columns:
                occurrence_data[presence_col] = 1
                self.logger.warning(
                    f"A coluna '{presence_col}' não foi encontrada. Ela foi criada com valores iguais a 1."
                )

            # Gerar dados de pseudoausência
            pseudo_ausencia_df = ModelDataPrepare().generate_pseudo_absence(
                occurrence_data,
                tiff_paths,
                n_pseudo_ausencias=int(len(occurrence_data) * pseudo_absence_ratio),
                presence_col=presence_col,
                lat_col=lat_col,
                lon_col=lon_col
            )

            # Marcar as pseudo-ausências como 0 e combinar com os dados de ocorrência
            pseudo_ausencia_df[presence_col] = 0
            occurrence_data = pd.concat([occurrence_data, pseudo_ausencia_df], ignore_index=True)

            # Preparar dados de raster e ocorrência
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)

            # Definir X (variáveis ambientais) e y (presença/ausência) para o modelo GLM
            X = raster_values
            y = occurrence_data[presence_col]

            # Ajustar o modelo GLM
            modelo = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            self.logger.info("Modelo GLM ajustado com sucesso.")

            # Preparar dados para previsão
            X_pred = matriz.reshape(-1, matriz.shape[2])
            nan_rows = np.isnan(X_pred).any(axis=1)  # Identificar linhas com NaN
            X_pred_valid = X_pred[~nan_rows]  # Remover linhas com NaN

            # Prever a distribuição
            previsao_valida = modelo.predict(X_pred_valid)

            # Criar um array completo de previsões com NaNs nas posições corretas
            previsao_glm = np.full(X_pred.shape[0], np.nan)
            previsao_glm[~nan_rows] = previsao_valida
            previsao_glm = previsao_glm.reshape(matriz.shape[0], matriz.shape[1])

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(previsao_glm, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return previsao_glm

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante a aplicação do GLM: {e}")
            raise
