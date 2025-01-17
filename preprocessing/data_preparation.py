# Funções para preparar dados raster e de ocorrência
import os
import rasterio
import numpy as np
import pandas as pd

from EcoDistrib.utils import LoggerManager, FileManager

class RasterDataExtract:
    def __init__(self):
        self.logger = LoggerManager().get_logger()

    def get_values(self, raster_path, coordinates, lat_col=None, lon_col=None, resolution=None, bounds=None, add_to_df=False):
        """
        Extrai valores de um arquivo raster ou de uma matriz com base em uma lista de coordenadas ou DataFrame.

        Parâmetros:
        - raster_path: Caminho do diretório contendo arquivos raster ou uma matriz de dados raster.
        - coordinates: Lista de tuplas com coordenadas (longitude, latitude) ou DataFrame com colunas de coordenadas.
        - lat_col: Nome da coluna de latitude no DataFrame (necessário se coordinates for DataFrame).
        - lon_col: Nome da coluna de longitude no DataFrame (necessário se coordinates for DataFrame).
        - resolution: Resolução dos dados (usado para matriz).
        - bounds: Limites dos dados (usado para matriz).
        - add_to_df: Booleano indicando se as novas colunas devem ser adicionadas ao DataFrame original,
                    uma para cada valor de pixel.

        Retorno:
        - DataFrame atualizado com novas colunas para cada raster, se add_to_df=True e coordinates for DataFrame.
        - Caso contrário, retorna uma lista de listas com valores de pixels para cada coordenada e raster.

        Exceções:
        - ValueError: Se colunas de latitude e longitude não forem especificadas quando coordinates for um DataFrame.
        """
        # Verifica se coordinates é um DataFrame e extrai as coordenadas, se necessário
        is_df = isinstance(coordinates, pd.DataFrame)
        if is_df:
            if lat_col is None or lon_col is None:
                self.logger.error("Colunas de latitude e longitude devem ser especificadas para um DataFrame.")
                raise ValueError("Especifique as colunas de latitude e longitude quando coordinates for um DataFrame.")
            coords_list = list(zip(coordinates[lon_col], coordinates[lat_col]))
        else:
            coords_list = coordinates

        # Obtém a lista de arquivos raster e seus nomes sem extensão
        try:
            raster_files = sorted(FileManager().listfile(raster_path))
            raster_names = [os.path.splitext(os.path.basename(file))[0] for file in raster_files]
        except Exception as e:
            self.logger.error(f"Erro ao listar os arquivos raster: {e}")
            raise

        # Obtém valores de raster ou matriz
        try:
            if isinstance(raster_path, np.ndarray):
                values_per_coordinate = self.get_matrix_values(raster_path, coords_list, resolution, bounds)
            else:
                values_per_coordinate = self.get_raster_values(raster_files, coords_list)
        except Exception as e:
            self.logger.error(f"Erro ao obter valores de raster ou matriz: {e}")
            raise

        # Adiciona os valores como novas colunas no DataFrame, se solicitado
        if add_to_df and is_df:
            for values, raster_name in zip(zip(*values_per_coordinate), raster_names):
                coordinates[raster_name] = values  # Usa o nome do arquivo como cabeçalho
            return coordinates

        return values_per_coordinate

    def get_raster_values(self, raster_path, coordinates):
        """
        Extrai valores de cada arquivo raster com base em uma lista de coordenadas.

        Parâmetros:
        - raster_path: Caminho do diretório contendo arquivos raster.
        - coordinates: Lista de tuplas com coordenadas (longitude, latitude).

        Retorno:
        - Lista de listas, onde cada sublista contém os valores correspondentes aos pixels nas coordenadas fornecidas,
        para cada raster.
        """
        values_per_coordinate = []

        try:
            # Obter lista de arquivos raster no diretório
            raster_files = sorted(FileManager().listfile(raster_path))

            for lat, lon in coordinates:
                values_for_current_coord = []

                for raster_file in raster_files:
                    try:
                        with rasterio.open(raster_file) as src:
                            # Converter coordenadas (lat, lon) para índice da linha e coluna no raster
                            row, col = src.index(lat, lon)

                            # Verificar se os índices estão dentro do intervalo do raster
                            if 0 <= row < src.height and 0 <= col < src.width:
                                # Ler o valor do pixel naquela linha e coluna (banda 1)
                                value = src.read(1)[row, col]
                                values_for_current_coord.append(value)
                            else:
                                values_for_current_coord.append(None)  # Coordenada fora da área do raster

                    except (IndexError, ValueError) as e:
                        # Tratamento para coordenadas fora do raster ou erros de índice
                        self.logger.error(f"Erro ao acessar coordenada ({lat}, {lon}) no arquivo {raster_file}: {e}")
                        values_for_current_coord.append(None)

                values_per_coordinate.append(values_for_current_coord)

        except Exception as e:
            self.logger.error(f"Erro ao processar arquivos raster no diretório '{raster_path}': {e}")
            raise

        return values_per_coordinate

    def get_matrix_values(self, raster_array, coordinates, resolution, bounds):
        """
        Extrai valores de uma matriz de dados raster com base em uma lista de coordenadas.

        Parâmetros:
        - raster_array (np.ndarray): Matriz de dados raster.
        - coordinates (list): Lista de tuplas com coordenadas (longitude, latitude).
        - resolution (tuple): Resolução da matriz no formato (res_x, res_y).
        - bounds (tuple): Limites da matriz no formato (min_lon, max_lon, min_lat, max_lat).

        Retorno:
        - list: Lista de valores correspondentes às coordenadas fornecidas.
        """
        values_per_coordinate = []

        # Extrair limites e resolução
        min_lon, max_lon, min_lat, max_lat = bounds
        res_x, res_y = resolution

        for lon, lat in coordinates:
            try:
                # Verificar se a coordenada está dentro dos limites
                if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                    # Calcular os índices na matriz
                    col = int((lon - min_lon) / res_x)  # Mapeia longitude para coluna
                    row = int((max_lat - lat) / res_y)  # Mapeia latitude para linha

                    # Verificar se os índices estão dentro do intervalo da matriz
                    if 0 <= row < raster_array.shape[0] and 0 <= col < raster_array.shape[1]:
                        value = raster_array[row, col]
                    else:
                        value = None  # Coordenada fora do intervalo da matriz
                else:
                    value = None  # Coordenada fora dos limites fornecidos

                values_per_coordinate.append(value)

            except Exception as e:
                self.logger.error(f"Erro ao processar coordenada ({lon}, {lat}): {e}")
                values_per_coordinate.append(None)

        return values_per_coordinate
