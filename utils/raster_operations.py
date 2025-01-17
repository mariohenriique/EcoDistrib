# Funções gerais de manipulação de raster

import os
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from rasterio.transform import from_origin

from EcoDistrib.utils.logger import LoggerManager
from EcoDistrib.utils.file_operations import FileManager

class RasterHandler:
    def __init__(self):
        self.logger = LoggerManager().get_logger()

    def crop_raster(self, raster_path: str, output_path: str = "", method: str = "bounding_box", bounding_box: list = None, shapefile_path: str = None) -> None:
        """
        Recorta um ou mais arquivos raster com base em um método especificado.

        Parâmetros:
        - raster_path (str): Caminho para um arquivo raster (.tif) ou uma pasta contendo arquivos raster.
        - output_path (str): Caminho para salvar os arquivos raster recortados. Se não especificado, usa o diretório atual.
        - method (str): Método de recorte a ser utilizado ("bounding_box", "polygon" ou "full").
        - bounding_box (list, opcional): Coordenadas da bounding box [min_lon, min_lat, max_lon, max_lat], necessário se `method` for "bounding_box".
        - shapefile_path (str, opcional): Caminho para o shapefile (polígono) a ser usado no recorte, necessário se `method` for "polygon".

        Exceções:
        - Levanta `ValueError` se o caminho especificado não for válido ou se os parâmetros do método forem inadequados.
        - Registra erros durante o processamento de arquivos individuais.
        """
        try:
            # Cria o diretório de saída, se não existir
            os.makedirs(output_path, exist_ok=True)

            # Verifica se o caminho é uma pasta ou um arquivo
            if os.path.isdir(raster_path):
                # Processa todos os arquivos .tif na pasta
                for filename in os.listdir(raster_path):
                    if filename.endswith('.tif'):
                        raster_file_path = os.path.join(raster_path, filename)
                        output_file_path = os.path.join(output_path, f"{os.path.splitext(filename)[0]}_recortado.tif")

                        try:
                            self.logger.info(f"Processando arquivo: {filename}")
                            self.crop_and_save(raster_file_path, output_file_path, method, bounding_box, shapefile_path)
                        except Exception as e:
                            self.logger.error(f"Erro ao processar {filename}: {e}")
            elif os.path.isfile(raster_path) and raster_path.endswith('.tif'):
                # Processa um único arquivo raster
                output_file_path = os.path.join(output_path, os.path.basename(raster_path))
                try:
                    self.crop_and_save(raster_path, output_file_path, method, bounding_box, shapefile_path)
                except Exception as e:
                    self.logger.error(f"Erro ao processar {raster_path}: {e}")
            else:
                raise ValueError("O caminho especificado deve ser um arquivo .tif ou uma pasta que contenha arquivos .tif.")

        except Exception as e:
            self.logger.error(f"Erro na execução de `recortar_raster`: {e}")

    def crop_raster_by_bounding_box(self, raster_path: str, output_path: str, bounding_box: list) -> None:
        """
        Recorta um raster com base em uma bounding box e salva o novo arquivo.

        Parâmetros:
        - raster_path (str): Caminho para o arquivo raster (.tif).
        - output_path (str): Caminho para salvar o raster recortado (.tif).
        - bounding_box (list): Coordenadas da bounding box [min_lon, min_lat, max_lon, max_lat].

        Exceções:
        - Levanta erros em caso de falhas no processamento, como arquivo inválido ou problemas de leitura/escrita.
        """
        try:
            self.logger.info(f"Iniciando recorte do raster {raster_path} com bounding box: {bounding_box}")

            # Abre o arquivo raster para leitura
            with rasterio.open(raster_path) as src:
                # Define a janela de recorte com base na bounding box
                window = src.window(*bounding_box)

                # Lê os dados do raster dentro da janela de recorte
                clipped_data = src.read(window=window)

                # Copia os metadados do raster original e atualiza com os novos valores
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": window.height,
                    "width": window.width,
                    "transform": src.window_transform(window)  # Atualiza a transformação do raster
                })

                # Salva o raster recortado utilizando a função 'salvar_raster'
                self.save_raster(output_path, clipped_data, out_meta)

            self.logger.info(f"Raster recortado por bounding box salvo em: {output_path}")

        except Exception as e:
            self.logger.error(f"Erro ao recortar o raster {raster_path} com bounding box {bounding_box}: {e}")
            raise

    def crop_raster_by_polygon(self, raster_path: str, output_path: str, shapefile_path: str) -> None:
        """
        Recorta um raster com base em um polígono (shapefile) e salva o novo arquivo raster no formato GeoTIFF.

        Parâmetros:
        - raster_path (str): Caminho para o arquivo raster de entrada (.tif).
        - output_path (str): Caminho para o arquivo raster de saída, onde o recorte será salvo (.tif).
        - shapefile_path (str): Caminho para o shapefile contendo o polígono de recorte.

        Comportamento:
        - Carrega o shapefile de polígono.
        - Aplica o recorte ao raster com base na geometria do shapefile.
        - Salva o raster recortado no formato GeoTIFF.
        - Registra mensagens de log sobre o progresso ou possíveis erros.

        Exceções:
        - FileNotFoundError: Quando o arquivo raster ou shapefile não é encontrado.
        - ValueError: Quando o shapefile não contém geometria válida.
        - Exception: Para outros erros inesperados.
        """
        try:
            self.logger.info(f"Iniciando recorte do raster {raster_path} usando o shapefile {shapefile_path}")

            # Abre o arquivo raster para leitura
            with rasterio.open(raster_path) as src:
                # Carrega o shapefile
                poligono = gpd.read_file(shapefile_path)

                # Verifica se o shapefile contém geometria válida
                if poligono.empty:
                    raise ValueError(f"Shapefile {shapefile_path} não contém geometria válida.")

                # Aplica o recorte com base no polígono
                clipped_data, out_transform = mask(src, poligono.geometry, crop=True)

                # Copia os metadados do raster original e atualiza com os novos valores
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": clipped_data.shape[1],
                    "width": clipped_data.shape[2],
                    "transform": out_transform
                })

                # Define valores fora da máscara como NaN
                clipped_data[clipped_data == src.nodata] = np.nan

                # Salva o novo raster recortado
                self.save_raster(output_path, clipped_data, out_meta)

            self.logger.info(f"Raster recortado por polígono salvo em: {output_path}")

        except FileNotFoundError as e:
            self.logger.error(f"Arquivo não encontrado: {e}")
            raise
        except ValueError as ve:
            self.logger.error(f"Erro de valor: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"Ocorreu um erro inesperado ao recortar o raster: {e}")
            raise

    def crop_whole_raster(self, raster_path: str, output_path: str) -> None:
        """
        Lê e salva o mapa inteiro sem realizar nenhum recorte.

        Parâmetros:
        - raster_path (str): Caminho para o arquivo raster de entrada (.tif).
        - output_path (str): Caminho para salvar o raster completo de saída (.tif).

        Comportamento:
        - Lê todos os dados do raster fornecido.
        - Copia os metadados do raster original.
        - Salva o raster no caminho de saída especificado.

        Exceções:
        - FileNotFoundError: Quando o arquivo raster não é encontrado.
        - Exception: Para outros erros inesperados.

        Logs:
        - Registra mensagens de sucesso, erro e progresso.
        """
        try:
            self.logger.info(f"Iniciando leitura do mapa inteiro de {raster_path}")

            # Abre o arquivo raster para leitura
            with rasterio.open(raster_path) as src:
                # Lê todos os dados do raster
                full_data = src.read()

                # Copia os metadados do raster original
                out_meta = src.meta.copy()

                # Salva o raster completo
                self.save_raster(output_path, full_data, out_meta)

            self.logger.info(f"Mapa inteiro salvo em: {output_path}")

        except FileNotFoundError:
            self.logger.error(f"Erro: Arquivo não encontrado - {raster_path}")
            raise
        except Exception as e:
            self.logger.error(f"Ocorreu um erro ao salvar o mapa inteiro: {e}")
            raise

    def save_raster(self, output_path: str, data: np.ndarray, profile: dict) -> None:
        """
        Salva um novo arquivo raster no caminho especificado.

        Parâmetros:
        - output_path (str): Caminho onde o novo arquivo raster será salvo.
        - data (np.ndarray): Array com os dados do raster a serem salvos.
        - profile (dict): Perfil/metadata do arquivo raster, incluindo informações de georreferenciamento, formato e dimensões.

        Comportamento:
        - Valida os parâmetros de entrada.
        - Salva o raster no formato GeoTIFF.

        Exceções:
        - ValueError: Se os dados forem vazios ou o perfil não for válido.
        - Exception: Para erros inesperados durante a escrita do arquivo.
        """
        try:
            # Validação dos dados
            if data is None or data.size == 0:
                raise ValueError("Os dados fornecidos para o raster estão vazios.")

            if not isinstance(profile, dict):
                raise ValueError("O perfil fornecido deve ser um dicionário com os metadados do raster.")

            # Atualiza o perfil para o driver GeoTIFF
            profile.update(driver='GTiff')

            # Salva o raster no arquivo especificado
            with rasterio.open(output_path, 'w', **profile) as dest:
                if len(data.shape) == 3:  # Múltiplas bandas
                    for i in range(data.shape[0]):
                        dest.write(data[i], i + 1)
                else:  # Única banda
                    dest.write(data, 1)

            self.logger.info(f"Novo arquivo raster salvo em: {output_path}")

        except ValueError as ve:
            self.logger.error(f"Erro de validação: {ve}")
            raise
        except Exception as e:
            self.logger.error(f"Ocorreu um erro ao salvar o raster: {e}")
            raise

    def crop_and_save(self, raster_path: str, output_path: str, method: str, bounding_box: list = None, shapefile_path: str = None) -> None:
        """
        Aplica o recorte em um arquivo raster com base no método escolhido e salva o resultado.

        Parâmetros:
        - raster_path (str): Caminho para o arquivo raster (.tif).
        - output_path (str): Caminho para salvar o raster recortado (.tif).
        - method (str): Método de recorte a ser utilizado ("bounding_box", "polygon" ou "full").
        - bounding_box (list, opcional): Coordenadas da bounding box [min_lon, min_lat, max_lon, max_lat], necessário se `method` for "bounding_box".
        - shapefile_path (str, opcional): Caminho para o shapefile, necessário se `method` for "polygon".

        Exceções:
        - Levanta `ValueError` se os parâmetros obrigatórios para o método selecionado não forem fornecidos.
        - Registra erros durante a execução.
        """
        try:
            self.logger.info(f"Aplicando recorte no arquivo: {raster_path} usando o método '{method}'.")

            if method == "bounding_box":
                if bounding_box is None:
                    raise ValueError("Bounding box deve ser fornecida se o método for 'bounding_box'")
                self.crop_raster_by_bounding_box(raster_path, output_path, bounding_box)

            elif method == "polygon":
                if shapefile_path is None:
                    raise ValueError("Caminho do shapefile deve ser fornecido se o método for 'polygon'")
                self.crop_raster_by_polygon(raster_path, output_path, shapefile_path)

            elif method == "full":
                self.crop_whole_raster(raster_path, output_path)

            else:
                raise ValueError("Método de recorte inválido. Os métodos possíveis são: 'bounding_box', 'polygon' ou 'full'.")

            self.logger.info(f"Recorte concluído e salvo em: {output_path}")

        except Exception as e:
            self.logger.error(f"Erro ao aplicar recorte no arquivo {raster_path}: {e}")
            raise

class RasterOperations:
    def __init__(self):
        self.logger = LoggerManager().get_logger()

    def raster_to_matrix(self,tiff_paths):
        """
        Converte os dados de vários arquivos TIFF em uma matriz 2D.

        Parâmetros:
        - tiff_paths (str ou list): Caminho para um diretório contendo arquivos TIFF ou uma lista de caminhos para arquivos TIFF a serem lidos.

        Retorno:
        - matriz (np.ndarray): Matriz 2D onde cada linha contém os dados de um raster.
        - nomes_variaveis (list): Lista com os nomes das variáveis (nomes dos arquivos sem extensão).

        Exceções:
        - ValueError: Se nenhum raster válido for encontrado.
        """
        # Lista os arquivos TIFF no caminho fornecido
        tiff_paths = FileManager.listfile(tiff_paths)

        # Inicializa listas para armazenar os dados e nomes das variáveis
        rasters_data = []
        nomes_variaveis = []

        for tiff in tiff_paths:
            try:
                with rasterio.open(tiff) as src:
                    # Lê a primeira banda e achata os dados 2D em 1D
                    data = src.read(1).flatten()

                    # Armazena os dados e o nome da variável (nome do arquivo sem extensão)
                    rasters_data.append(data)
                    nome_variavel = os.path.splitext(os.path.basename(tiff))[0]
                    nomes_variaveis.append(nome_variavel)
            except FileNotFoundError:
                self.logger.error(f"Erro: Arquivo não encontrado - {tiff}")
            except rasterio.errors.RasterioError as re:
                self.logger.error(f"Erro ao processar o arquivo {tiff}: {re}")
            except Exception as e:
                self.logger.error(f"Erro inesperado ao ler o arquivo {tiff}: {e}")

        # Valida se há dados coletados
        if not rasters_data:
            raise ValueError("Nenhum raster válido foi encontrado.")

        # Converte os dados coletados em uma matriz e remove colunas com NaN
        matriz = np.array(rasters_data)
        matriz = self.remove_pixels_nan(matriz)

        return matriz, nomes_variaveis

    def raster_to_matrix_2d(self,tiff_paths):
        """
        Converte os dados de vários arquivos TIFF em uma matriz 3D.

        Parâmetros:
        - tiff_paths (str ou list): Caminho para um diretório contendo arquivos TIFF ou uma lista de caminhos para arquivos TIFF a serem lidos.

        Retorno:
        - matriz (np.ndarray): Matriz 3D onde cada camada corresponde a um raster.
        - nomes_variaveis (list): Lista com os nomes das variáveis (nomes dos arquivos sem extensão).
        - resolution (tuple): Resolução dos rasters (res_x, res_y).
        - bounds (tuple): Limites da matriz (min_lon, max_lon, min_lat, max_lat).

        Exceções:
        - ValueError: Se nenhum raster válido for encontrado.
        """
        # Lista os arquivos TIFF no caminho fornecido
        tiff_paths = FileManager().listfile(tiff_paths)

        # Inicializa listas e variáveis para armazenamento
        rasters_data = []
        nomes_variaveis = []
        min_lon, max_lon, min_lat, max_lat = float('inf'), float('-inf'), float('inf'), float('-inf')
        res_x, res_y = None, None

        for tiff in tiff_paths:
            try:
                with rasterio.open(tiff) as src:
                    # Lê a primeira banda do raster e armazena os dados
                    data = src.read(1)
                    rasters_data.append(data)
                    nomes_variaveis.append(os.path.splitext(os.path.basename(tiff))[0])

                    # Atualiza resolução e limites geográficos
                    if res_x is None and res_y is None:
                        res_x, res_y = src.res

                    bounds = src.bounds
                    min_lon = min(min_lon, bounds.left)
                    max_lon = max(max_lon, bounds.right)
                    min_lat = min(min_lat, bounds.bottom)
                    max_lat = max(max_lat, bounds.top)

            except FileNotFoundError:
                self.logger.error(f"Erro: Arquivo não encontrado - {tiff}")
            except rasterio.errors.RasterioError as re:
                self.logger.error(f"Erro ao processar o arquivo {tiff}: {re}")
            except Exception as e:
                self.logger.error(f"Erro inesperado ao ler o arquivo {tiff}: {e}")

        # Verifica se há dados coletados
        if not rasters_data:
            raise ValueError("Nenhum raster válido foi encontrado.")

        # Converte a lista de rasters em uma matriz 3D (cada camada é um raster)
        matriz = np.array(rasters_data)

        # Ajusta a ordem dos eixos se necessário
        matriz = np.moveaxis(matriz, 0, -1)  # Move o eixo do raster para ser o último

        # Define os limites e a resolução como tuplas
        bounds_tuple = (min_lon, max_lon, min_lat, max_lat)
        resolution_tuple = (res_x, res_y)

        return matriz, nomes_variaveis, resolution_tuple, bounds_tuple

    def remove_pixels_nan(self,matriz):
        """
        Remove as colunas que possuem apenas valores NaN em todos os mapas (linhas) da matriz.

        Parâmetros:
        - matriz (np.ndarray): Matriz onde cada linha contém os dados de um raster.

        Retorno:
        - matriz_filtrada (np.ndarray): Matriz com as colunas que tinham apenas NaN removidas.

        Exceções:
        - ValueError: Se a matriz fornecida estiver vazia ou não for bidimensional.
        """
        # Verifica se a matriz não está vazia
        if matriz.size == 0:
            raise ValueError("A matriz fornecida está vazia.")

        # Cria uma máscara para identificar colunas válidas
        valid_columns = ~np.isnan(matriz).all(axis=0)

        # Aplica a máscara para filtrar a matriz
        matriz_filtrada = matriz[:, valid_columns]

        return matriz_filtrada

# Verificar um jeito melhor de fazer isso de forma que possa converter qualquer arquivo para outro
class RasterConverter:
    def __init__(self):
        self.logger = LoggerManager().get_logger()

    def convert_csv_to_tif(self, csv_file):
        """
        Converte um arquivo CSV contendo coordenadas e valores da coluna 'mean' para GeoTIFF (.tif),
        e exclui o arquivo .csv. Caso não encontre a coluna 'mean' ou as colunas de coordenadas, registra
        um aviso e não processa o arquivo.

        Parâmetros:
        - csv_file (str): Caminho para o arquivo .csv a ser convertido.

        Exceções tratadas:
        - FileNotFoundError: Caso o arquivo CSV não exista.
        - Exception: Erros gerais durante a leitura ou conversão.
        """
        try:
            # Define o nome do arquivo .tif
            tif_file = csv_file.replace('.csv', '.tif')

            # Carregar dados do CSV
            df = pd.read_csv(csv_file)

            # Tenta identificar automaticamente as colunas de coordenadas e valores
            col_names = df.columns

            # Procurar por colunas relacionadas a coordenadas e valores
            x_col = next((col for col in col_names if 'lon' in col.lower() or 'x' in col.lower()), None)
            y_col = next((col for col in col_names if 'lat' in col.lower() or 'y' in col.lower()), None)
            value_col = next((col for col in col_names if 'mean' in col.lower()), None)

            # Verificar se a coluna 'mean' foi encontrada
            if not value_col:
                self.logger.warning(f"A coluna 'mean' não foi encontrada no arquivo {csv_file}. Este arquivo não será processado.")
                return

            # Verificar se as colunas de coordenadas foram identificadas
            if not (x_col and y_col):
                self.logger.warning(f"Não foi possível identificar as colunas de coordenadas no arquivo {csv_file}.")
                return

            # Extrair coordenadas e valores
            x_coords = df[x_col].values
            y_coords = df[y_col].values
            values = df[value_col].values

            # Criar uma grade vazia para preencher com os valores
            grid_shape = (len(np.unique(y_coords)), len(np.unique(x_coords)))
            data = np.full(grid_shape, np.nan)

            for x, y, value in zip(x_coords, y_coords, values):
                x_index = np.where(np.unique(x_coords) == x)[0][0]
                y_index = np.where(np.unique(y_coords) == y)[0][0]
                data[y_index, x_index] = value

            # Configurar a transformação de coordenadas
            transform = from_origin(x_coords.min(), y_coords.max(), abs(x_coords[1] - x_coords[0]), abs(y_coords[1] - y_coords[0]))

            # Salvar o arquivo .tif
            with rasterio.open(tif_file, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype='float32', crs='EPSG:4326', transform=transform) as dst:
                dst.write(data, 1)

            self.logger.info(f"Arquivo convertido para GeoTIFF: {tif_file}")

            # Excluir o arquivo .csv
            FileManager.exclude_file(csv_file)

        except FileNotFoundError:
            self.logger.error(f"Arquivo CSV '{csv_file}' não encontrado.")
        except Exception as e:
            self.logger.error(f"Erro durante a conversão do arquivo '{csv_file}' para GeoTIFF: {e}")


    def convert_asc_to_tif(self, asc_file):
        """
        Converte um arquivo ASC para o formato GeoTIFF (TIF), criando um arquivo TIF para cada camada
        caso o arquivo tenha múltiplas camadas.

        Parâmetros:
        - asc_file (str): Caminho do arquivo ASC a ser convertido.

        Logs:
        - Registra eventos de sucesso ou falha durante a conversão.
        """
        # Define o prefixo para os arquivos .tif
        tif_prefix = asc_file.replace('.asc', '_layer_')

        try:
            # Abre o arquivo .asc e lê as informações para conversão
            with rasterio.open(asc_file) as src:
                # Obtém o perfil e os dados do arquivo
                profile = src.profile
                data = src.read()

                self.logger.info(f"Shape dos dados lidos: {data.shape}")

                # Verifica se há mais de uma camada
                if len(data.shape) == 3:
                    # Para cada camada no arquivo, cria um arquivo .tif separado
                    for i, layer in enumerate(data, start=1):
                        # Atualiza o perfil para o formato de saída como GeoTIFF
                        profile.update(driver='GTiff', count=1)

                        # Nome do arquivo para a camada
                        tif_file = f"{tif_prefix}{i}.tif"

                        # Salva a camada no arquivo .tif
                        with rasterio.open(tif_file, 'w', **profile) as dst:
                            dst.write(layer, 1)

                        self.logger.info(f"Camada {i} salva com sucesso como {tif_file}")

                else:
                    # Se houver apenas uma camada, salva a mesma camada
                    tif_file = f"{tif_prefix}.tif"
                    profile.update(driver='GTiff', count=1)

                    with rasterio.open(tif_file, 'w', **profile) as dst:
                        dst.write(data[0], 1)

                    self.logger.info(f"Arquivo convertido para GeoTIFF: {tif_file}")

        except Exception as e:
            self.logger.error(f"Erro ao converter o arquivo '{asc_file}' para GeoTIFF: {e}")

    def convert_tif_to_asc(self, input_dir, output_dir, nodata_value=-9999):
        """
        Converte arquivos .tif em um diretório para .asc e os salva em um diretório de saída,
        substituindo valores NaN pelo valor especificado para nodata.

        Parâmetros:
        - input_dir (str): Diretório contendo os arquivos .tif.
        - output_dir (str): Diretório onde os arquivos .asc serão salvos.
        - nodata_value (float ou int, opcional): Valor a ser usado para nodata. O padrão é -9999.
        """

        # Verifica se o diretório de saída existe, caso contrário, cria
        os.makedirs(output_dir, exist_ok=True)

        # Lista todos os arquivos .tif no diretório de entrada
        for file in os.listdir(input_dir):
            if file.endswith(".tif"):
                tif_path = os.path.join(input_dir, file)

                # Nome do arquivo de saída com extensão .asc
                asc_filename = os.path.splitext(file)[0] + ".asc"
                asc_path = os.path.join(output_dir, asc_filename)

                try:
                    # Abre o arquivo .tif usando rasterio
                    with rasterio.open(tif_path) as dataset:
                        # Lê os dados como um array e substitui NaN pelo valor nodata_value
                        array = dataset.read(1)

                        # Substitui valores NaN pelo valor nodata_value
                        array = np.where(np.isnan(array), nodata_value, array)

                        # Abre um novo arquivo .asc para escrita
                        with open(asc_path, 'w') as asc_file:
                            # Escreve o cabeçalho do arquivo .asc
                            asc_file.write(f"ncols        {dataset.width}\n")
                            asc_file.write(f"nrows        {dataset.height}\n")
                            asc_file.write(f"xllcorner    {dataset.bounds.left}\n")
                            asc_file.write(f"yllcorner    {dataset.bounds.bottom}\n")
                            asc_file.write(f"cellsize     {dataset.res[0]}\n")
                            asc_file.write(f"NODATA_value {nodata_value}\n")
                            
                            # Escreve os dados no formato .asc
                            for row in array:
                                asc_file.write(' '.join(map(str, row)) + '\n')

                        self.logger.info(f"Arquivo convertido com sucesso: {asc_path}")
                except Exception as e:
                    self.logger.error(f"Erro ao processar o arquivo {tif_path}: {str(e)}")
            else:
                self.logger.warning(f"Arquivo ignorado (não é .tif): {file}")
