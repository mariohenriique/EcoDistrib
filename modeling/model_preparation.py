import random
import numpy as np
import pandas as pd
from scipy.stats import mode

from EcoDistrib.common import msg_logger
from EcoDistrib.outputs import MapGenerator
from EcoDistrib.utils import FileManager, RasterOperations
from EcoDistrib.preprocessing import RasterDataExtract

class ModelDataPrepare:
    def __init__(self):
        self.logger = msg_logger

    def prepare_raster_data(
            self,
            tiff_paths,
            occurrence_data,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            formato='GTiff'
        ):
        """
        Prepara os dados de raster e ocorrência para cálculo de distâncias.

        Parâmetros:
        - tiff_paths (str ou list): Caminho para um diretório contendo arquivos TIFF ou uma lista de caminhos para arquivos TIFF.
        - occurrence_data (pd.DataFrame ou list): Dados de ocorrência contendo coordenadas (latitude e longitude).
        - lat_col (str, opcional): Nome da coluna de latitude no DataFrame de ocorrência. Padrão: 'decimalLatitude'.
        - lon_col (str, opcional): Nome da coluna de longitude no DataFrame de ocorrência. Padrão: 'decimalLongitude'.

        Retorna:
        - matriz (np.ndarray): Matriz 3D com valores das camadas de raster (dimensões: camadas x linhas x colunas).
        - raster_values (np.ndarray): Matriz 2D com valores extraídos para cada coordenada de ocorrência (dimensões: pontos x camadas).
        - profile (dict): Perfil do raster sintético para referência futura.

        Logs:
        - Registra mensagens informando o progresso e eventuais erros.

        Erros Tratados:
        - Registra e retorna erros relacionados ao carregamento de TIFFs ou extração de valores.
        """
        try:
            # Criar o perfil do raster sintético
            _, profile = MapGenerator().create_synthetic_raster(formato=formato)

            # Obter a lista de arquivos TIFF
            tiff_files = FileManager().listfile(tiff_paths)
            self.logger.info(f"{len(tiff_files)} arquivos TIFF encontrados para processamento.")

            if not tiff_files:
                self.logger.error("Nenhum arquivo TIFF encontrado nos caminhos fornecidos.")
                raise FileNotFoundError("Nenhum arquivo TIFF foi localizado no diretório ou na lista fornecida.")

            # Converter rasters para matriz 3D
            matriz, _, __, ___ = RasterOperations().raster_to_matrix_2d(tiff_files)
            self.logger.info(f"Matriz 3D de rasters gerada com dimensões: {matriz.shape}.")

            # Validar os dados de ocorrência
            if occurrence_data is None or len(occurrence_data) == 0:
                self.logger.error("Os dados de ocorrência estão vazios ou inválidos.")
                raise ValueError("Os dados de ocorrência não podem ser vazios.")

            # Obter valores de raster para cada coordenada usando a função get_values
            values_per_coordinate = RasterDataExtract().get_values(tiff_files, occurrence_data, lat_col=lat_col, lon_col=lon_col)
            self.logger.info("Valores de raster extraídos para as coordenadas de ocorrência.")

            # Extrair valores para cada coordenada e organizar em uma matriz 2D
            raster_values = []
            for raster_idx, _ in enumerate(tiff_files):
                raster_value = [coord_values[raster_idx] for coord_values in values_per_coordinate]
                raster_values.append(raster_value)

            # Converter a lista de listas em matriz 2D
            raster_values = np.array(raster_values).T  # Cada linha representa uma coordenada
            self.logger.info(f"Matriz 2D de valores extraídos gerada com dimensões: {raster_values.shape}.")

            return matriz, raster_values, profile

        except Exception as e:
            self.logger.error(f"Erro ao preparar os dados raster: {e}")
            raise

    def calculate_central_point(self,raster_values, method='mean'):
        """
        Calcula o ponto central com base no método escolhido pelo usuário.

        Parâmetros:
        - raster_values (np.ndarray): 
            Array 2D onde cada linha representa os valores de uma coordenada em cada camada.
        - method (str): 
            Método para calcular o ponto central. Pode ser 'media', 'mediana' ou 'moda'.

        Retorno:
        - np.ndarray: O ponto central calculado com base no método escolhido.

        Erros:
        - Levanta ValueError se o método for desconhecido.
        - Registra logs para informar erros ou sucesso na execução.

        Logs:
        - Registra o método utilizado e a conclusão da operação.
        - Registra erro caso o método fornecido seja inválido.
        """
        try:
            if method == 'mean':
                result = np.mean(raster_values, axis=0)
            elif method == 'median':
                result = np.median(raster_values, axis=0)
            elif method == 'mode':
                # Calcula a moda ignorando valores NaN
                result = mode(raster_values, axis=0, nan_policy='omit').mode[0]
            else:
                raise ValueError("Método desconhecido para ponto central. Escolha entre 'media', 'mediana', ou 'moda'.")

            self.logger.info(f"Ponto central calculado usando o método '{method}'.")
            return result

        except ValueError as e:
            self.logger.error(f"Erro: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado ao calcular o ponto central: {e}")
            raise

    def generate_pseudo_absence(
            self,
            occurrence_data, 
            tiff_paths, 
            n_pseudo_ausencias=None, 
            lat_col='decimalLatitude', 
            lon_col='decimalLongitude', 
            presence_col='presence'
        ):
        """
        Gera pontos de pseudo-ausência aleatoriamente dentro dos limites do raster.

        Parâmetros:
        - occurrence_data (pd.DataFrame): 
            Dados de ocorrência, contendo as colunas de latitude e longitude.
        - tiff_paths (str ou list): 
            Caminho para os arquivos TIFF com dados ambientais.
        - n_pseudo_ausencias (int, opcional): 
            Número de pontos de pseudo-ausência a ser gerado. Se não especificado, será 30% do tamanho de `occurrence_data`.
        - lat_col (str, opcional): 
            Nome da coluna com a latitude no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional): 
            Nome da coluna com a longitude no DataFrame (padrão: 'decimalLongitude').
        - presence_col (str, opcional): 
            Nome da coluna indicando presença ou ausência (padrão: 'presence').

        Retorno:
        - pd.DataFrame: 
            DataFrame com as coordenadas de pseudo-ausência e a coluna de presença configurada para 0.

        Logs:
        - Informa o progresso na geração de pseudo-ausências.
        - Registra mensagens de erro em caso de falhas.
        """
        try:
            # Obter a lista de arquivos TIFF
            tif_files = FileManager().listfile(tiff_paths)
            if not tif_files:
                raise ValueError("Nenhum arquivo TIFF encontrado no caminho fornecido.")

            # Definir o número de pseudo-ausências
            if not n_pseudo_ausencias:
                n_pseudo_ausencias = int(len(occurrence_data) * 0.30)
                self.logger.info(f"Número de pseudo-ausências não especificado. Usando 30% das ocorrências: {n_pseudo_ausencias}")

            # Preparar os limites do raster
            _, _, profile = self.prepare_raster_data(tif_files, occurrence_data, lat_col, lon_col)
            transform = profile['transform']

            # Obter os limites geográficos do raster
            min_lon, max_lat = transform * (0, 0)
            max_lon, min_lat = transform * (profile['width'], profile['height'])

            pseudo_ausencias = []
            while len(pseudo_ausencias) < n_pseudo_ausencias:
                # Gerar coordenadas aleatórias dentro da extensão
                random_lon = random.uniform(min_lon, max_lon)
                random_lat = random.uniform(min_lat, max_lat)

                # Obter valores de raster para as coordenadas
                values_per_coordinate = RasterDataExtract().get_values(
                    coordinates=[[random_lon, random_lat]], 
                    raster_path=tif_files
                )
                values_per_coordinate = np.array(values_per_coordinate, dtype=np.float64)

                # Validar a ausência e verificar NaN
                if not any(
                    np.isclose(occurrence_data[lat_col], random_lat) & 
                    np.isclose(occurrence_data[lon_col], random_lon)
                ) and not np.any(np.isnan(values_per_coordinate)):
                    pseudo_ausencias.append([random_lat, random_lon])

            # Criar DataFrame com os pontos de pseudo-ausência
            pseudo_ausencia_df = pd.DataFrame(pseudo_ausencias, columns=[lat_col, lon_col])
            pseudo_ausencia_df[presence_col] = 0  # Marcar como ausência

            self.logger.info(f"Pseudo-ausências geradas com sucesso: {len(pseudo_ausencias)} pontos.")
            return pseudo_ausencia_df

        except ValueError as ve:
            self.logger.error(f"Erro de validação: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado ao gerar pseudo-ausências: {e}")
            raise
