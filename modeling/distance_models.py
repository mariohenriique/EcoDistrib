# Funções de modelagem baseadas em distâncias
import os
import rasterio
import numpy as np
from scipy.spatial.distance import mahalanobis, canberra, chebyshev, cosine, minkowski

from EcoDistrib.outputs import MapGenerator
from EcoDistrib.utils import FileManager
from EcoDistrib.preprocessing import RasterDataExtract
from EcoDistrib.modeling import ModelDataPrepare
from EcoDistrib.common import msg_logger

class DistanceModeling:
    def __init__(self):
        self.logger = msg_logger
        self.model_type = None

    def sdm_bioclim(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_bioclim.tif',
        ):
        """
        Aplica o algoritmo Bioclim para modelagem de distribuição de espécies.

        Parâmetros:
        - occurrence_data (pd.DataFrame ou list):
            Dados contendo coordenadas de ocorrência (latitude e longitude).
        - tiff_paths (str ou list):
            Caminho para um diretório ou lista de arquivos TIFF contendo as variáveis ambientais.
        - lat_col (str, opcional):
            Nome da coluna com as latitudes no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional):
            Nome da coluna com as longitudes no DataFrame (padrão: 'decimalLongitude').
        - save (bool, opcional):
            Indica se o mapa resultante deve ser salvo como arquivo TIFF (padrão: False).
        - output_save (str, opcional):
            Caminho para salvar o mapa resultante se `save=True` (padrão: 'mapa_resultante_bioclim.tif').

        Retorno:
        - np.ndarray:
            Array normalizado com os resultados do modelo Bioclim.

        Logs:
        - Informações de progresso, como carregamento de dados e sucesso do processamento.
        - Erros críticos, como arquivos TIFF ausentes ou dados inválidos.
        """
        self.model_type = 'Bioclim'
        try:
            # Validar e carregar os arquivos TIFF
            tiff_files = FileManager().listfile(tiff_paths)
            threshold=len(tiff_files)
            if not tiff_files:
                raise ValueError("Nenhum arquivo TIFF encontrado no caminho fornecido.")

            self.logger.info(f"{len(tiff_files)} arquivos TIFF encontrados para processamento.")

            # Criar o raster sintético e obter o perfil
            _, profile = MapGenerator().create_synthetic_raster(formato=formato)

            # Obter valores ambientais para cada coordenada
            values_per_coordinate = RasterDataExtract().get_values(
                tiff_files, 
                occurrence_data, 
                lat_col=lat_col, 
                lon_col=lon_col
            )
            self.logger.info("Valores ambientais extraídos para as coordenadas de ocorrência.")

            result_arrays = []

            # Iterar sobre cada camada raster
            for raster_idx, raster_file in enumerate(tiff_files):
                # Obter valores de ocorrência para esta camada
                raster_values = [coord_values[raster_idx] for coord_values in values_per_coordinate]

                # Calcular os limites (min e max)
                min_value = np.nanmin(raster_values)
                max_value = np.nanmax(raster_values)

                # Ler os dados do raster
                with rasterio.open(raster_file) as src:
                    raster_data = src.read(1)

                # Criar array de resultado com NaN como padrão
                result_array = np.full_like(raster_data, np.nan, dtype=float)

                # Máscara para valores dentro e fora do intervalo min-max
                mask_within_range = (raster_data >= min_value) & (raster_data <= max_value)
                mask_outside_range = ~mask_within_range & np.isfinite(raster_data)

                # Aplicar as máscaras ao array de resultado
                result_array[mask_within_range] = 1
                result_array[mask_outside_range] = 0

                result_arrays.append(result_array)

            # Somar os arrays resultantes
            final_result_array = np.sum(result_arrays, axis=0)

            # Aplicar a normalização baseada no threshold
            nan_mask = np.isnan(final_result_array)

            final_result_array_threshold = np.where(final_result_array >= threshold,1,0).astype(float)
            final_result_array_threshold[nan_mask] = np.nan

            final_result_array = (final_result_array) / (threshold)  # Normaliza para [0, 1]
            self.logger.info("Mapa resultante processado com sucesso.")

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(final_result_array, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return final_result_array

        except ValueError as ve:
            self.logger.error(f"Erro de validação: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado ao aplicar o algoritmo Bioclim: {e}")
            raise

    def sdm_mahalanobis(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            central_point_method='mean',
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_mahalanobis.tif'
        ):
        """
        Calcula a distância de Mahalanobis para um conjunto de dados de ocorrência.

        Parâmetros:
        - occurrence_data (pd.DataFrame ou list):
            Dados contendo coordenadas de ocorrência (latitude e longitude).
        - tiff_paths (str ou list):
            Caminho para um diretório ou uma lista de caminhos para arquivos TIFF.
        - central_point_method (str, opcional):
            Método para calcular o ponto central ('media', 'mediana' ou 'moda').
        - lat_col (str, opcional):
            Nome da coluna de latitude no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional):
            Nome da coluna de longitude no DataFrame (padrão: 'decimalLongitude').
        - save (bool, opcional):
            Indica se o resultado deve ser salvo em arquivo TIFF (padrão: False).
        - output_save (str, opcional):
            Caminho para salvar o arquivo de saída, se `save=True` (padrão: 'mapa_resultante_mahalanobis.tif').

        Retorno:
        - np.ndarray:
            Array 2D com as distâncias de Mahalanobis para cada ponto do raster.
        
        Logs:
        - Informações de progresso e possíveis erros durante o processamento.
        """
        self.model_type = 'Mahalanobis'
        try:
            # Preparar os dados dos rasters
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col,formato)
            self.logger.info("Dados de raster preparados com sucesso.")

            # Calcular a matriz de covariância e sua inversa
            cov_matrix = np.cov(raster_values, rowvar=False)
            VI = np.linalg.inv(cov_matrix)
            self.logger.info("Matriz de covariância e inversa calculadas.")

            # Calcular o ponto central com o método especificado
            ponto_central = ModelDataPrepare().calculate_central_point(raster_values, central_point_method)
            self.logger.info(f"Ponto central calculado usando o método '{central_point_method}'.")

            # Inicializar o array de distâncias
            n_lat, n_lon, n_layers = matriz.shape
            distancias_array = np.full((n_lat, n_lon), np.nan, dtype=float)  # Inicializa com NaN

            # Calcular a distância de Mahalanobis para cada ponto no raster
            for lat_idx in range(n_lat):
                for lon_idx in range(n_lon):
                    ponto = matriz[lat_idx, lon_idx, :]  # Extrair valores das camadas para o ponto

                    if not np.any(np.isnan(ponto)):  # Verificar se o ponto contém apenas valores válidos
                        distancias_array[lat_idx, lon_idx] = mahalanobis(ponto, ponto_central, VI)

            self.logger.info("Distâncias de Mahalanobis calculadas para todos os pontos.")

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(distancias_array, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return distancias_array

        except np.linalg.LinAlgError as lae:
            self.logger.error(f"Erro na inversão da matriz de covariância: {lae}")
            raise

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante o cálculo de Mahalanobis: {e}")
            raise

    def sdm_manhattan(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            central_point_method='mean',
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_manhattan.tif'
        ):
        """
        Calcula a distância de Manhattan para um conjunto de dados de ocorrência.

        Parâmetros:
        - occurrence_data (pd.DataFrame ou list):
            Dados contendo coordenadas de ocorrência (latitude e longitude).
        - tiff_paths (str ou list):
            Caminho para um diretório ou uma lista de caminhos para arquivos TIFF.
        - central_point_method (str, opcional):
            Método para calcular o ponto central ('media', 'mediana' ou 'moda').
        - lat_col (str, opcional):
            Nome da coluna de latitude no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional):
            Nome da coluna de longitude no DataFrame (padrão: 'decimalLongitude').
        - save (bool, opcional):
            Indica se o resultado deve ser salvo em arquivo TIFF (padrão: False).
        - output_save (str, opcional):
            Caminho para salvar o arquivo de saída, se `save=True` (padrão: 'mapa_resultante_manhattan.tif').

        Retorno:
        - np.ndarray:
            Array 2D com as distâncias de Manhattan para cada ponto do raster.
        
        Logs:
        - Mensagens de progresso e erros são registrados usando `self.logger`.
        """
        self.model_type = 'Manhattan'
        try:
            # Preparar os dados dos rasters
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col,formato)
            self.logger.info("Dados de raster preparados com sucesso.")

            # Calcular o ponto central com o método especificado
            ponto_central = ModelDataPrepare().calculate_central_point(raster_values, central_point_method)
            self.logger.info(f"Ponto central calculado usando o método '{central_point_method}'.")

            # Inicializar o array de distâncias
            n_lat, n_lon, n_layers = matriz.shape
            distancias_array = np.full((n_lat, n_lon), np.nan, dtype=float)  # Inicializa com NaN

            # Calcular a distância de Manhattan para cada ponto no raster
            for lat_idx in range(n_lat):
                for lon_idx in range(n_lon):
                    ponto = matriz[lat_idx, lon_idx, :]  # Extrair valores das camadas para o ponto

                    if not np.any(np.isnan(ponto)):  # Verificar se o ponto contém apenas valores válidos
                        distancias_array[lat_idx, lon_idx] = np.sum(np.abs(ponto - ponto_central))

            self.logger.info("Distâncias de Manhattan calculadas para todos os pontos.")

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(distancias_array, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return distancias_array

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante o cálculo de Manhattan: {e}")
            raise

    def sdm_euclidean(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            central_point_method='mean',  # Método para calcular o ponto central
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_euclidean.tif'
        ):
        """
        Calcula a distância euclidiana para cada ponto do raster em relação ao ponto central.

        Parâmetros:
        - occurrence_data (pd.DataFrame ou list):
            Dados contendo coordenadas de ocorrência (latitude e longitude).
        - tiff_paths (str ou list):
            Caminho para um diretório ou uma lista de caminhos para arquivos TIFF.
        - central_point_method (str, opcional):
            Método para calcular o ponto central ('media', 'mediana' ou 'moda').
        - lat_col (str, opcional):
            Nome da coluna de latitude no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional):
            Nome da coluna de longitude no DataFrame (padrão: 'decimalLongitude').
        - save (bool, opcional):
            Indica se o resultado deve ser salvo em arquivo TIFF (padrão: False).
        - output_save (str, opcional):
            Caminho para salvar o arquivo de saída, se `save=True` (padrão: 'mapa_resultante_euclidean.tif').

        Retorno:
        - np.ndarray:
            Array 2D com as distâncias euclidianas para cada ponto do raster.

        Logs:
        - Mensagens de progresso e erros são registrados usando `self.logger`.
        """
        self.model_type = 'Euclidean'
        try:
            # Preparar os dados dos rasters
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)
            self.logger.info("Dados de raster preparados com sucesso.")

            # Calcular o ponto central com o método especificado
            ponto_central = ModelDataPrepare().calculate_central_point(raster_values, central_point_method)
            self.logger.info(f"Ponto central calculado usando o método '{central_point_method}'.")

            dists_euclidean = []

            # Inicializar o array de distâncias
            n_lat, n_lon, n_layers = matriz.shape
            # distancias_array = np.full((n_lat, n_lon), np.nan, dtype=float)  # Inicializa com NaN

            # Calcular a distância euclidiana para cada ponto no raster
            for lat_idx in range(n_lat):
                for lon_idx in range(n_lon):
                    ponto = matriz[lat_idx, lon_idx, :]  # Extrair valores das camadas para o ponto

                    # Verificar se o ponto tem valores NaN
                    if np.any(np.isnan(ponto)):  # Se qualquer valor do ponto for NaN
                        dist = np.nan  # Atribuir NaN para a distância
                    else:
                        dist = np.linalg.norm(ponto - ponto_central)  # Calcular a distância Euclidiana
                    dists_euclidean.append(dist)

            distancias_array = np.array(dists_euclidean).reshape(n_lat, n_lon)
            self.logger.info("Distâncias euclidianas calculadas para todos os pontos.")

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(distancias_array, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return distancias_array

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante o cálculo de distância euclidiana: {e}")
            raise

    def sdm_canberra(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            central_point_method='mean',  # Método para calcular o ponto central
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_canberra.tif'
        ):
        """
        Calcula a distância de Canberra para cada ponto do raster em relação ao ponto central.

        Parâmetros:
        - occurrence_data (pd.DataFrame ou list):
            Dados contendo coordenadas de ocorrência (latitude e longitude).
        - tiff_paths (str ou list):
            Caminho para um diretório ou uma lista de caminhos para arquivos TIFF.
        - central_point_method (str, opcional):
            Método para calcular o ponto central ('media', 'mediana' ou 'moda').
        - lat_col (str, opcional):
            Nome da coluna de latitude no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional):
            Nome da coluna de longitude no DataFrame (padrão: 'decimalLongitude').
        - save (bool, opcional):
            Indica se o resultado deve ser salvo em arquivo TIFF (padrão: False).
        - output_save (str, opcional):
            Caminho para salvar o arquivo de saída, se `save=True` (padrão: 'mapa_resultante_canberra.tif').

        Retorno:
        - np.ndarray:
            Array 2D com as distâncias de Canberra para cada ponto do raster.

        Logs:
        - Mensagens de progresso e erros são registrados usando `self.logger`.
        """
        self.model_type = 'Canberra'
        try:
            # Preparar os dados dos rasters
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)
            self.logger.info("Dados de raster preparados com sucesso.")

            # Calcular o ponto central com o método especificado
            ponto_central = ModelDataPrepare().calculate_central_point(raster_values, central_point_method)
            self.logger.info(f"Ponto central calculado usando o método '{central_point_method}'.")
            # Inicializar lista para armazenar as distâncias de Canberra
            dists_canberra = []
            # Inicializar o array de distâncias
            n_lat, n_lon, n_layers = matriz.shape

            # Calcular a distância de Canberra para cada ponto no raster
            for lat_idx in range(n_lat):
                for lon_idx in range(n_lon):
                    ponto = matriz[lat_idx, lon_idx, :]  # Extrair valores das camadas para o ponto

                    if np.any(np.isnan(ponto)):  # Verificar valores NaN
                        dist = np.nan
                    else:
                        dist = canberra(ponto, ponto_central)  # Calcular distância de Canberra

                    dists_canberra.append(dist)

            distancias_array = np.array(dists_canberra).reshape(n_lat, n_lon)
            self.logger.info("Distâncias de Canberra calculadas para todos os pontos.")

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(distancias_array, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return distancias_array

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante o cálculo de distância de Canberra: {e}")
            raise

    def sdm_chebyshev(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            central_point_method='mean',
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_chebyshev.tif'
        ):
        """
        Calcula a distância de Chebyshev para um conjunto de dados de ocorrência.

        Parâmetros:
        - occurrence_data (pd.DataFrame ou list):
            Dados contendo coordenadas de ocorrência (latitude e longitude).
        - tiff_paths (str ou list):
            Caminho para um diretório ou uma lista de caminhos para arquivos TIFF.
        - central_point_method (str, opcional):
            Método para calcular o ponto central ('media', 'mediana' ou 'moda').
        - lat_col (str, opcional):
            Nome da coluna de latitude no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional):
            Nome da coluna de longitude no DataFrame (padrão: 'decimalLongitude').
        - save (bool, opcional):
            Indica se o resultado deve ser salvo em arquivo TIFF (padrão: False).
        - output_save (str, opcional):
            Caminho para salvar o arquivo de saída, se `save=True` (padrão: 'mapa_resultante_chebyshev.tif').

        Retorno:
        - np.ndarray:
            Array 2D com as distâncias de Chebyshev para cada ponto do raster.

        Logs:
        - Mensagens de progresso e erros são registrados usando `self.logger`.
        """
        self.model_type = 'Chebyshev'
        try:
            # Preparar os dados dos rasters
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)
            self.logger.info("Dados de raster preparados com sucesso.")

            # Calcular o ponto central com o método especificado
            ponto_central = ModelDataPrepare().calculate_central_point(raster_values, central_point_method)
            self.logger.info(f"Ponto central calculado usando o método '{central_point_method}'.")

            # Inicializar lista para armazenar as distâncias de Chebyshev
            dists_chebyshev = []

            # Obter a forma da matriz 3D (latitude, longitude, camadas)
            n_lat, n_lon, n_layers = matriz.shape

            # Calcular a distância de Chebyshev para cada ponto no raster
            for lat_idx in range(n_lat):
                for lon_idx in range(n_lon):
                    ponto = matriz[lat_idx, lon_idx, :]  # Extrair os valores das camadas para o ponto

                    if np.any(np.isnan(ponto)):  # Verificar valores NaN
                        dist = np.nan
                    else:
                        dist = chebyshev(ponto, ponto_central)  # Calcular distância de Chebyshev

                    dists_chebyshev.append(dist)

            # Converter a lista de distâncias em um array 2D com a forma correta
            distancias_array = np.array(dists_chebyshev).reshape(n_lat, n_lon)

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(distancias_array, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return distancias_array

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante o cálculo de distância de Chebyshev: {e}")
            raise

    def sdm_cosseno(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            central_point_method='mean',
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_cosseno.tif'
        ):
        """
        Calcula a distância do Cosseno para um conjunto de dados de ocorrência.

        Parâmetros:
        - occurrence_data (pd.DataFrame ou list):
            Dados contendo coordenadas de ocorrência (latitude e longitude).
        - tiff_paths (str ou list):
            Caminho para um diretório ou uma lista de caminhos para arquivos TIFF.
        - central_point_method (str, opcional):
            Método para calcular o ponto central ('media', 'mediana' ou 'moda').
        - lat_col (str, opcional):
            Nome da coluna de latitude no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional):
            Nome da coluna de longitude no DataFrame (padrão: 'decimalLongitude').
        - save (bool, opcional):
            Indica se o resultado deve ser salvo em arquivo TIFF (padrão: False).
        - output_save (str, opcional):
            Caminho para salvar o arquivo de saída, se `save=True` (padrão: 'mapa_resultante_cosseno.tif').

        Retorno:
        - np.ndarray:
            Array 2D com as distâncias do Cosseno para cada ponto do raster.

        Logs:
        - Mensagens de progresso e erros são registrados usando `self.logger`.
        """
        self.model_type = 'Cosseno'
        try:
            # Preparar os dados dos rasters
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)
            self.logger.info("Dados de raster preparados com sucesso.")

            # Calcular o ponto central com o método especificado
            ponto_central = ModelDataPrepare().calculate_central_point(raster_values, central_point_method)
            self.logger.info(f"Ponto central calculado usando o método '{central_point_method}'.")

            # Inicializar lista para armazenar as distâncias do Cosseno
            dists_cosseno = []

            # Obter a forma da matriz 3D (latitude, longitude, camadas)
            n_lat, n_lon, n_layers = matriz.shape

            # Calcular a distância do Cosseno para cada ponto no raster
            for lat_idx in range(n_lat):
                for lon_idx in range(n_lon):
                    ponto = matriz[lat_idx, lon_idx, :]  # Extrair os valores das camadas para o ponto

                    if np.any(np.isnan(ponto)):  # Verificar valores NaN
                        dist = np.nan
                    else:
                        dist = cosine(ponto, ponto_central)  # Calcular distância do Cosseno

                    dists_cosseno.append(dist)

            # Converter a lista de distâncias em um array 2D com a forma correta
            distancias_array = np.array(dists_cosseno).reshape(n_lat, n_lon)

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(distancias_array, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return distancias_array

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante o cálculo de distância do Cosseno: {e}")
            raise

    def sdm_minkowski(
            self,
            occurrence_data,
            tiff_paths,
            lat_col='decimalLatitude',
            lon_col='decimalLongitude',
            central_point_method='mean',
            p=3,  # Parâmetro de Minkowski (p=1 é Manhattan, p=2 é Euclidiana, p>2 é generalizado)
            save=False,
            formato='GTiff',
            output_save='mapa_resultante_minkowski.tif'
        ):
        """
        Calcula a distância Minkowski para um conjunto de dados de ocorrência.

        Parâmetros:
        - occurrence_data (pd.DataFrame ou list):
            Dados contendo coordenadas de ocorrência (latitude e longitude).
        - tiff_paths (str ou list):
            Caminho para um diretório ou uma lista de caminhos para arquivos TIFF.
        - central_point_method (str, opcional):
            Método para calcular o ponto central ('media', 'mediana' ou 'moda').
        - p (float, opcional):
            Parâmetro da distância de Minkowski (p=1 é Manhattan, p=2 é Euclidiana, valores maiores generalizam).
        - lat_col (str, opcional):
            Nome da coluna de latitude no DataFrame (padrão: 'decimalLatitude').
        - lon_col (str, opcional):
            Nome da coluna de longitude no DataFrame (padrão: 'decimalLongitude').
        - save (bool, opcional):
            Indica se o resultado deve ser salvo em arquivo TIFF (padrão: False).
        - output_save (str, opcional):
            Caminho para salvar o arquivo de saída, se `save=True` (padrão: 'mapa_resultante_minkowski.tif').

        Retorno:
        - np.ndarray:
            Array 2D com as distâncias de Minkowski para cada ponto do raster.

        Logs:
        - Mensagens de progresso e erros são registrados usando `self.logger`.
        """
        self.model_type = 'Minkowski'
        try:
            # Preparar os dados dos rasters
            matriz, raster_values, profile = ModelDataPrepare().prepare_raster_data(tiff_paths, occurrence_data, lat_col, lon_col, formato)
            self.logger.info("Dados de raster preparados com sucesso.")

            # Calcular o ponto central com o método especificado
            ponto_central = ModelDataPrepare().calculate_central_point(raster_values, central_point_method)
            self.logger.info(f"Ponto central calculado usando o método '{central_point_method}'.")

            # Inicializar lista para armazenar as distâncias de Minkowski
            dists_minkowski = []

            # Obter a forma da matriz 3D (latitude, longitude, camadas)
            n_lat, n_lon, n_layers = matriz.shape

            # Calcular a distância de Minkowski para cada ponto no raster
            for lat_idx in range(n_lat):
                for lon_idx in range(n_lon):
                    ponto = matriz[lat_idx, lon_idx, :]  # Extrair os valores das camadas para o ponto

                    if np.any(np.isnan(ponto)):  # Verificar valores NaN
                        dist = np.nan
                    else:
                        dist = minkowski(ponto, ponto_central, p)  # Calcular distância de Minkowski

                    dists_minkowski.append(dist)

            # Converter a lista de distâncias em um array 2D com a forma correta
            distancias_array = np.array(dists_minkowski).reshape(n_lat, n_lon)

            # Salvar o resultado se solicitado
            if save:
                # MapGenerator.salvar_mapa(final_result_array_threshold, profile, output_save=output_save)
                MapGenerator().save_map(distancias_array, profile, output_save=output_save)
                self.logger.info(f"Mapa resultante salvo em: {output_save}")

            return distancias_array

        except ValueError as ve:
            self.logger.error(f"Erro de validação nos dados: {ve}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado durante o cálculo de distância de Minkowski: {e}")
            raise
