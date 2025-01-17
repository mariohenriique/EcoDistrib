# Funções para manipular e criar shapefiles
import os
import zipfile
import requests
import tempfile
import geopandas as gpd

from EcoDistrib.common import msg_logger
# from EcoDistrib.utils import LoggerManager

class ShapefileHandler:
    def __init__(self):
        self.logger = msg_logger

    def download_shapefile_natural_earth(self, url, caminho_zip):
        """
        Baixa e extrai um shapefile do Natural Earth.

        Parâmetros:
        - url (str): URL para o arquivo ZIP do shapefile.
        - caminho_zip (str): Caminho completo para salvar o arquivo ZIP baixado (inclui o nome do arquivo ZIP).

        Retorno:
        - None: O conteúdo do ZIP será extraído no mesmo diretório onde o arquivo ZIP foi salvo.
        """
        try:
            # Baixar o arquivo ZIP
            resposta = requests.get(url, stream=True)
            resposta.raise_for_status()  # Levanta um erro para status HTTP inválidos

            # Salvar o arquivo ZIP no caminho especificado
            with open(caminho_zip, 'wb') as arquivo_zip:
                for chunk in resposta.iter_content(chunk_size=1024):
                    arquivo_zip.write(chunk)

            print(f"Arquivo ZIP baixado com sucesso: {caminho_zip}")

            # Extrair o conteúdo do ZIP
            with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(caminho_zip))
                print(f"Shapefile extraído em: {os.path.dirname(caminho_zip)}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro ao baixar o shapefile: {e}")
            raise

        except zipfile.BadZipFile:
            self.logger.error("O arquivo baixado não é um ZIP válido.")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado: {e}")
            raise

    def create_shapefile_countries(self, nomes_paises, output_path):
        """
        Cria um shapefile para os países especificados.

        Parâmetros:
        - nomes_paises (list): Lista de nomes dos países a serem incluídos no shapefile.
        - output_path (str): Caminho para salvar o shapefile filtrado.
        """
        try:
            # URL do shapefile do Natural Earth
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

            # Criar uma pasta temporária
            with tempfile.TemporaryDirectory() as temp_dir:
                caminho_zip = os.path.join(temp_dir, "naturalearth_countries.zip")

                # Baixar e extrair o shapefile
                self.download_shapefile_natural_earth(url, caminho_zip)

                # Caminho para o shapefile extraído
                shapefile_path = os.path.join(temp_dir, "ne_110m_admin_0_countries.shp")

                # Carregar os dados de países do Natural Earth
                world = gpd.read_file(shapefile_path)

                # Filtrar o GeoDataFrame pelos nomes dos países
                paises_filtrados = world[world['ADMIN'].isin(nomes_paises)]

                if paises_filtrados.empty:
                    self.logger.error(f"Nenhum país encontrado para os nomes: {nomes_paises}")
                    return

                # Criar o diretório de saída caso ele não exista
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Salvar o shapefile dos países filtrados
                paises_filtrados.to_file(output_path)
                self.logger.info(f"Shapefile criado para os países: {nomes_paises} em: {output_path}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro ao baixar o shapefile: {e}")
            raise

        except FileNotFoundError as e:
            self.logger.error(f"Arquivo shapefile não encontrado: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado: {e}")
            raise

    def download_shapefile_brazilian_state(self, url, caminho_zip):
        """
        Baixa e extrai um shapefile dos estados do Brasil.

        Parâmetros:
        - url (str): URL para o arquivo ZIP do shapefile.
        - caminho_zip (str): Caminho para salvar o arquivo ZIP baixado.
        """
        try:
            # Baixar o arquivo ZIP
            resposta = requests.get(url)

            # Verificar se a resposta foi bem-sucedida
            if resposta.status_code == 200:
                with open(caminho_zip, 'wb') as arquivo_zip:
                    arquivo_zip.write(resposta.content)

                # Extrair o conteúdo do ZIP
                with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(caminho_zip))

                self.logger.info(f"Shapefile dos estados do Brasil baixado e extraído com sucesso para: {caminho_zip}")
            else:
                self.logger.error(f"Erro ao baixar o arquivo: {resposta.status_code} - {resposta.text}")
                raise Exception(f"Erro ao baixar o arquivo: {resposta.status_code}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro ao fazer o download do shapefile: {e}")
            raise

        except zipfile.BadZipFile as e:
            self.logger.error(f"Erro ao extrair o arquivo ZIP: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Erro inesperado: {e}")
            raise

    def create_shapefile_states(self, siglas_estados, output_path):
        """
        Cria um shapefile para os estados brasileiros especificados.

        Parâmetros:
        - siglas_estados (list): Lista de siglas dos estados a serem incluídos no shapefile.
        - output_path (str): Caminho para salvar o shapefile.
        """
        # URL do shapefile dos estados do Brasil
        url = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/Brasil/BR/BR_UF_2022.zip"

        try:
            # Criar uma pasta temporária
            with tempfile.TemporaryDirectory() as temp_dir:
                caminho_zip = os.path.join(temp_dir, "estados_brasil.zip")

                # Baixar e extrair o shapefile
                self.download_shapefile_brazilian_state(url, caminho_zip)

                # Carregar os dados dos estados do Brasil
                shapefile_path = os.path.join(temp_dir, "BR_UF_2022.shp")  # Ajuste o nome do arquivo conforme necessário
                estados_brasil = gpd.read_file(shapefile_path)

                # Verifique as colunas disponíveis (opcional para conferir a coluna correta)
                self.logger.info(f"Colunas disponíveis no shapefile: {estados_brasil.columns}")

                # Filtrar o GeoDataFrame pelas siglas dos estados
                estados_filtrados = estados_brasil[estados_brasil['SIGLA_UF'].isin(siglas_estados)]

                if estados_filtrados.empty:
                    self.logger.warning(f"Nenhum estado encontrado para as siglas: {siglas_estados}")
                    return

                # Salvar o shapefile dos estados
                estados_filtrados.to_file(output_path)
                self.logger.info(f"Shapefile criado para os estados: {siglas_estados} em: {output_path}")

        except Exception as e:
            self.logger.error(f"Erro ao criar o shapefile: {e}")
            raise
