# Funções de PCA
import os
import rasterio
import numpy as np
from sklearn.decomposition import PCA

from EcoDistrib.common import msg_logger
from EcoDistrib.utils import FileManager
from EcoDistrib.outputs import MapGenerator

class PCAProcessor:
    def __init__(self):
        self.logger = msg_logger

    def apply_pca(self, input_folder, output_folder="pca_components", n_components=3):
        """
        Aplica Análise de Componentes Principais (PCA) a um conjunto de arquivos raster e salva os componentes principais.

        Parâmetros:
        - input_folder (str): Diretório contendo os arquivos raster de entrada (TIFFs).
        - output_folder (str): Diretório onde os componentes principais serão salvos. Padrão: "pca_components".
        - n_components (int): Número de componentes principais a serem gerados. Padrão: 3.

        Retorno:
        - None: Os componentes principais são salvos como arquivos raster no diretório de saída.
        """
        try:
            # Criar o diretório de saída, se não existir
            os.makedirs(output_folder, exist_ok=True)

            # 1. Listar arquivos TIFF no diretório de entrada
            tiff_paths = FileManager.listfile(input_folder)
            if not tiff_paths:
                raise ValueError(f"Nenhum arquivo TIFF encontrado no diretório: {input_folder}")

            # 2. Ler cada arquivo TIFF e armazenar seus dados
            rasters_data = []
            valid_masks = []  # Lista para armazenar as máscaras de validade
            height, width = None, None

            for tiff in tiff_paths:
                with rasterio.open(tiff) as src:
                    data = src.read(1)  # Lê a primeira banda
                    if height is None or width is None:
                        height, width = data.shape

                    # Cria uma máscara de validade (True para valores válidos, False para NaN)
                    mask = ~np.isnan(data)
                    valid_masks.append(mask)
                    rasters_data.append(data)

            # Empilhar os dados em uma matriz 3D
            matriz_3d = np.array(rasters_data)

            # Achatar a matriz 3D e criar uma máscara global
            matriz_flat = matriz_3d.reshape(matriz_3d.shape[0], -1)
            global_mask = np.logical_and.reduce(valid_masks)  # Combina todas as máscaras

            # 3. Remover colunas com NaNs usando a máscara global
            matriz_sem_nan = matriz_flat[:, global_mask.flatten()]
            if matriz_sem_nan.shape[1] == 0:
                raise ValueError("Todos os pixels válidos foram removidos devido a NaNs nos rasters.")

            # 4. Aplicar PCA
            pca = PCA(n_components=n_components)
            transformed_data = pca.fit_transform(matriz_sem_nan.T)  # Transpor para amostras como linhas

            # 5. Criar uma matriz para armazenar os componentes principais
            _, profile = MapGenerator().create_synthetic_raster(raster_shape=(height, width))
            pca_components = np.full((n_components, height * width), np.nan)  # Preenche com NaN inicialmente
            pca_components[:, global_mask.flatten()] = transformed_data.T

            # 6. Salvar os componentes principais como arquivos raster
            for i in range(n_components):
                component = pca_components[i].reshape((height, width))
                # Ajustar para o usuário escolher o nome dos arquivos
                output_path = os.path.join(output_folder, f'pca_component_{i + 1}.tif')
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(component, 1)

            print(f"PCA realizado com sucesso! Componentes salvos em: {output_folder}")

        except Exception as e:
            self.logger.error(f"Erro ao aplicar PCA: {e}")
            raise
