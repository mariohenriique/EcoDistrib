# Funções para geração e salvamento de mapas
import rasterio
import numpy as np
from rasterio.transform import from_origin

from EcoDistrib.common import msg_logger

class MapGenerator:
    def __init__(self):
        self.logger = msg_logger

    def save_map(self, array, profile, output_save=''):
        """
        Salva um array NumPy como um arquivo raster no formato .tif.

        Parâmetros:
        - array (numpy.ndarray): Array com os valores a serem salvos.
        - profile (dict): Dicionário contendo as configurações do raster, como CRS, resolução, e transformações.
        - output_save (str, opcional): Caminho para salvar o arquivo raster. O padrão é 'mapa_resultante_bioclim.tif'.

        Retorno:
        - Nenhum.

        Logs:
        - Registra mensagens informando sucesso ou falhas ao salvar o mapa.

        Erros Tratados:
        - Caso o arquivo não possa ser salvo, o erro será registrado com uma mensagem descritiva.
        """
        try:
            with rasterio.open(output_save, 'w', **profile) as dst:
                dst.write(array, 1)
            self.logger.info(f"Mapa salvo com sucesso em: {output_save}")
        except Exception as e:
            self.logger.error(f"Erro ao salvar o mapa em {output_save}: {e}")

    def create_synthetic_raster(
            self,
            formato='GTiff',
            raster_shape=(937, 943),
            resolution=(0.04167, 0.04167),
            origin=(-74.0, 5.25),
            nodata_value=-3.3999999521443642e+38,
            crs='EPSG:4326',
            initial_value=np.nan
            ):
        """
        Cria um raster sintético inicializado com valores especificados, retornando o array e seu perfil geoespacial.

        Parâmetros:
        - raster_shape (tuple, opcional): Dimensão do raster (linhas, colunas).
        Padrão ajustado para 2.5 minutos de resolução sobre o Brasil.
        - resolution (tuple, opcional): Resolução espacial do raster em graus (tamanho do pixel em x e y).
        Padrão é 0.04167 (~2.5 minutos).
        - origin (tuple, opcional): Coordenada de origem no canto superior esquerdo do raster (longitude, latitude).
        Padrão é a esquina noroeste do Brasil.
        - nodata_value (float, opcional): Valor nodata para áreas sem dados. Padrão: -3.3999999521443642e+38.
        - crs (str, opcional): Sistema de referência espacial. Padrão: 'EPSG:4326' (WGS84).
        - initial_value (int ou float, opcional): Valor inicial para os pixels do raster. Padrão: NaN.

        Retorno:
        - tuple:
            - array (numpy.ndarray): Array inicializado com os valores especificados.
            - profile (dict): Dicionário com metadados do raster, necessário para salvamento ou manipulação futura.

        Logs:
        - Registra mensagens informando sucesso na criação do raster.

        Erros Tratados:
        - Nenhum erro explícito tratado nesta função.
        """
        try:
            # Inicializar o array com o valor inicial especificado
            raster_array = np.full(raster_shape, initial_value, dtype='float32')

            # Definir a transformação geoespacial e o perfil do raster
            transform = from_origin(origin[0], origin[1], resolution[0], resolution[1])
            profile = {
                'driver': 'GTiff',
                'dtype': 'float32',
                'nodata': nodata_value,
                'width': raster_shape[1],
                'height': raster_shape[0],
                'count': 1,
                'crs': crs,
                'transform': transform
            }

            self.logger.info(f"Raster sintético criado com tamanho {raster_shape}, resolução {resolution}, origem {origin}.")
            return raster_array, profile

        except Exception as e:
            self.logger.error(f"Erro ao criar o raster sintético: {e}")
            raise
