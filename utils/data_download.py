# Funções para download de dados
import os
import requests
import pyo_oracle
from bs4 import BeautifulSoup

from EcoDistrib.utils import FileManager
from EcoDistrib.utils import RasterConverter


class DataDownloader:
    def __init__(self):
        from EcoDistrib.common import msg_logger
        self.logger = msg_logger


    def download_data_wordclim(self, output_dir=None):
        """
        Baixa arquivos ZIP do WorldClim, os extrai e exclui os arquivos ZIP.

        Parâmetros:
        - output_dir (str, opcional): Diretório onde os dados extraídos serão armazenados.
        Caso não seja fornecido, os dados serão salvos no diretório padrão 'downloaded_data/wordclim_data'.

        Exceções tratadas:
        - requests.exceptions.RequestException: Erros relacionados ao download dos arquivos.
        - Outras exceções genéricas serão capturadas e registradas no logger.
        """
        # Define o diretório onde os dados serão armazenados
        output_dir = os.path.join(output_dir, "downloaded_data", "wordclim_data") if output_dir else "downloaded_data/wordclim_data"

        # Lista de URLs dos arquivos .zip do WorldClim
        urls = [
            "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip",
            # "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_elev.zip",
        ]

        # Cria o diretório, se não existir
        os.makedirs(output_dir, exist_ok=True)

        # Itera sobre a lista de URLs e processa cada uma
        for url in urls:
            # Nome do arquivo ZIP para salvar
            output_file = os.path.join(output_dir, os.path.basename(url))

            try:
                # Log de início do download
                self.logger.info(f"Iniciando download dos dados do WorldClim de {url}...")

                # Baixa o arquivo
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Verifica se o download foi bem-sucedido

                # Salva o arquivo ZIP
                with open(output_file, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=128):
                        file.write(chunk)

                # Log de sucesso do download
                self.logger.info(f"Download concluído. Dados salvos em: {output_file}")

                # Extrai o arquivo ZIP e exclui o original
                FileManager().extract_exclude_zip_file(output_file=output_file, output_dir=output_dir)

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Erro ao baixar o arquivo de {url}: {e}")
            except Exception as e:
                self.logger.error(f"Erro inesperado ao processar o arquivo de {url}: {e}")

    def download_data_biooracle(self, output_dir=None):
        """
        Baixa dados do BioOracle em formato CSV para o diretório especificado.

        Parâmetros:
        - output_dir (str, opcional): Diretório onde os dados extraídos serão armazenados.
        Caso não fornecido, o diretório padrão será "downloaded_data/biooracle_data".

        Exceções tratadas:
        - Exception: Erros gerais durante o download ou manipulação dos dados.
        """
        # Define o diretório onde os dados serão armazenados
        output_dir = os.path.join(output_dir, "downloaded_data", "biooracle_data") if output_dir else "downloaded_data/biooracle_data"

        # Obtém os IDs dos conjuntos de dados disponíveis para o período atual
        try:
            dataset_ids = pyo_oracle.list_layers(time_period='present')
            self.logger.info("IDs dos datasets obtidos com sucesso para o período 'present'.")
        except Exception as e:
            self.logger.error(f"Erro ao obter os IDs dos datasets do BioOracle: {e}")
            return

        # Define as restrições para o download das camadas
        constraints = {
            "latitude>=": 0,
            "latitude<=": 10,
            "latitude_step": 100,
            "longitude>=": 0,
            "longitude<=": 10,
            "longitude_step": 1
        }

        # Cria o diretório, se não existir
        os.makedirs(output_dir, exist_ok=True)

        # Faz o download das camadas do BioOracle
        try:
            self.logger.info("Iniciando o download das camadas do BioOracle.")

            downloaded_files = pyo_oracle.download_layers(
                dataset_ids.datasetID,
                output_directory=output_dir,
                response='csv',
                constraints=constraints,
                skip_confirmation=True
            )

            self.logger.info(f"Dados do BioOracle baixados com sucesso em: {output_dir}")
            
            # Possível conversão de CSV para TIFF (comentada por enquanto)
            # for file in os.listdir(output_dir):
            #     if file.endswith('.csv'):
            #         csv_file = os.path.join(output_dir, file)
            #         self.convert_csv_to_tif(csv_file)  # Exemplo de função para conversão

        except Exception as e:
            self.logger.error(f"Erro ao baixar dados do BioOracle: {e}")

    def download_data_earthenv(self, output_dir=None):
        """
        Baixa dados do EarthEnv e os salva em um diretório especificado.

        Parâmetros:
        - output_dir (str): Diretório onde os dados extraídos serão armazenados.
        Se não fornecido, o diretório padrão será "downloaded_data/earthenv_data".

        Logs:
        - Registra eventos como sucesso no download, falhas em requisições e criação de diretórios.
        """
        # Define o diretório onde os dados serão armazenados
        output_dir = os.path.join(output_dir, "downloaded_data", "earthenv_data") if output_dir else "downloaded_data/earthenv_data"

        # Extrai as URLs da tabela 'landcoverfull' no EarthEnv
        try:
            urls = self.extract_urls_from_table_landcoverfull()
            if not urls:
                self.logger.warning("Nenhuma URL foi encontrada na tabela 'landcoverfull'. Processo abortado.")
                return
        except Exception as e:
            self.logger.error(f"Erro ao extrair URLs da tabela 'landcoverfull': {e}")
            return

        # Cria o diretório, se não existir
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Diretório criado ou já existente: {output_dir}")
        except Exception as e:
            self.logger.error(f"Erro ao criar o diretório '{output_dir}': {e}")
            return

        # Baixa os arquivos listados nas URLs
        for url in urls:
            # Nome do arquivo para salvar
            output_file = os.path.join(output_dir, os.path.basename(url))

            try:
                # Baixar o arquivo
                self.logger.info(f"Baixando dados do EarthEnv de {url}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Verifica se o download foi bem-sucedido

                # Salvar o arquivo
                with open(output_file, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=128):
                        file.write(chunk)

                self.logger.info(f"Download concluído. Dados salvos em: {output_file}")

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Erro ao baixar o arquivo de {url}: {e}")
            except Exception as e:
                self.logger.error(f"Erro inesperado ao processar a URL '{url}': {e}")

    def download_data_ufz(self, output_dir=None):
        """
        Baixa dados do UFZ, converte arquivos .asc para .tif e exclui os arquivos .asc.

        Parâmetros:
        - output_dir (str): Diretório onde os dados extraídos serão armazenados.
        Se não fornecido, o diretório padrão será "downloaded_data/ufz_data".

        Logs:
        - Registra eventos de sucesso ou falha para cada etapa do processo.
        """
        # Define o diretório onde os dados serão armazenados
        output_dir = os.path.join(output_dir, "downloaded_data", "ufz_data") if output_dir else "downloaded_data/ufz_data"

        base_url = "https://www.ufz.de"
        page_url = "https://www.ufz.de/gluv/index.php?en=32435"

        # Cria o diretório de saída se não existir
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Diretório criado ou já existente: {output_dir}")
        except Exception as e:
            self.logger.error(f"Erro ao criar o diretório '{output_dir}': {e}")
            return

        try:
            # Faz a requisição HTTP para a página principal
            response = requests.get(page_url)
            response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

            # Analisa o HTML com BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Encontra as URLs dos arquivos .asc
            asc_urls = [base_url + link['href'] for link in soup.find_all('a', href=True) if link['href'].endswith('.asc')]

            if not asc_urls:
                self.logger.warning("Nenhuma URL de arquivo .asc foi encontrada na página.")
                return

            self.logger.info(f"{len(asc_urls)} arquivos .asc encontrados para download.")

            # Baixa e processa os arquivos .asc
            for asc_url in asc_urls:
                # Ignora URLs contendo 'monthly' conforme o critério definido
                if 'monthly' in asc_url:
                    self.logger.info(f"Ignorando arquivo mensal: {asc_url}")
                    continue

                file_name = os.path.basename(asc_url)
                asc_file = os.path.join(output_dir, file_name)

                try:
                    # Download do arquivo .asc
                    self.logger.info(f"Baixando arquivo: {asc_url}")
                    asc_response = requests.get(asc_url)
                    asc_response.raise_for_status()

                    # Salva o arquivo .asc
                    with open(asc_file, 'wb') as file:
                        file.write(asc_response.content)
                    self.logger.info(f"Download concluído: {asc_file}")

                    # Converte o arquivo .asc para .tif
                    RasterConverter().convert_asc_to_tif(asc_file)

                    # Remove o arquivo .asc
                    FileManager().exclude_file(asc_file)

                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Erro ao baixar ou processar o arquivo '{asc_url}': {e}")
                except Exception as e:
                    self.logger.error(f"Erro inesperado ao processar o arquivo '{asc_url}': {e}")

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro ao acessar a página '{page_url}': {e}")
        except Exception as e:
            self.logger.error(f"Erro inesperado ao acessar ou processar dados do UFZ: {e}")

    def download_data(self,output_dir="", destination_dir=None):
        """
        Apresenta um menu interativo para o usuário escolher se deseja baixar dados
        de diferentes fontes ou apenas alguns específicos.

        Parâmetros:
        - output_dir (str): Diretório onde os dados baixados serão armazenados.
        - destination_dir (str): Diretório onde os arquivos .tif serão armazenados.

        Comportamento:
        - Exibe um menu de opções para que o usuário escolha baixar dados de diferentes fontes (WorldClim, BioOracle, EarthEnv, UFZ).
        - O usuário pode optar por baixar todos os dados de uma vez ou escolher individualmente quais fontes deseja baixar.
        - Se o usuário escolher baixar todos os dados, a função chamará cada função correspondente para baixar os dados.
        - Se o usuário optar por selecionar as fontes, ele deve inserir os números correspondentes às fontes desejadas.
        - Após os downloads, se `destination_dir` for especificado, a função `move_tif_files` será chamada para mover os arquivos .tif para o diretório de destino.

        Exceções:
        - A função pode lançar erros se houver problemas com as requisições de download ou se o usuário fornecer uma entrada inválida para as opções.
        - Não há validação sobre o formato do diretório especificado em `output_dir` ou `destination_dir`, o que pode causar erros se os diretórios não existirem.
        """
        # Lista de funções de download
        functions = {
            1: ("WorldClim", self.download_data_wordclim),
            2: ("BioOracle", self.download_data_biooracle),
            3: ("EarthEnv", self.download_data_earthenv),
            4: ("UFZ", self.download_data_ufz)
        }

        print("\n### Menu de Downloads ###")

        # Perguntar se o usuário deseja baixar todos os dados ou escolher individualmente
        download_all = input("Você gostaria de baixar todos os dados disponíveis de uma vez? (s/n): ").strip().lower()

        if download_all == 's':
            # Baixar todos os dados
            try:
                for name, func in functions.values():
                    print(f"Iniciando download dos dados de {name}...")
                    func(output_dir)
            except Exception as e:
                # Registra erro caso ocorra alguma falha durante o download
                self.logger.error(f"Erro ao baixar todos os dados: {e}")
        else:
            # Exibir opções de fontes de dados
            print("\nSelecione os dados que deseja baixar:")
            for key, (name, _) in functions.items():
                print(f"{key}. {name}")

            # Entrada do usuário
            choices = input("\nDigite os números correspondentes às fontes que deseja baixar (separados por vírgula) ou '0' para todos: ").strip().split(',')


            # Verificar se o usuário escolheu '0' para baixar todos os dados
            if '0' in choices:
                try:
                    for name, func in functions.values():
                        print(f"Iniciando download dos dados de {name}...")
                        func(output_dir)
                except Exception as e:
                    # Registra erro caso ocorra alguma falha durante o download
                    self.logger.error(f"Erro ao baixar todos os dados após escolha '0': {e}")
            else:
                # Baixar apenas os dados selecionados
                for choice in choices:
                    choice = choice.strip()  # Remover espaços em branco
                    if choice.isdigit() and int(choice) in functions:
                        name, func = functions[int(choice)]
                        try:
                            print(f"Iniciando download dos dados de {name}...")
                            func(output_dir)
                        except Exception as e:
                            # Registra erro caso ocorra alguma falha ao baixar dados de uma fonte específica
                            self.logger.error(f"Erro ao baixar dados de {name}: {e}")
                    else:
                        print(f"Escolha inválida: {choice}")

        # Movimentação de arquivos .tif, se 'destination_dir' foi passado
        if destination_dir:
            try:
                FileManager().move_tif_files(destination_dir, output_dir)
            except Exception as e:
                # Registra erro caso ocorra alguma falha na movimentação
                self.logger.error(f"Erro ao mover arquivos .tif para '{destination_dir}': {e}")

    def extract_urls_from_table_landcoverfull(self):
        """
        Faz a requisição para uma página, localiza uma tabela com o ID 'landcoverfull'
        e extrai todas as URLs contidas nas tags <a> dentro da tabela.

        Retorna:
        - List[str]: Lista de URLs extraídas da tabela ou uma lista vazia em caso de erro.

        Logs:
        - Registra erros de requisição ou caso a tabela não seja encontrada.
        """
        url = "https://www.earthenv.org/landcover"

        try:
            # Fazer a requisição HTTP
            response = requests.get(url)
            response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

            # Analisar o HTML com BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Encontrar a tabela com o ID 'landcoverfull'
            table = soup.find('table', id='landcoverfull')

            # Extrair todas as tags <a> dentro da tabela, se a tabela foi encontrada
            if table:
                links = table.find_all('a')
                # Criar uma lista para armazenar as URLs
                urls = [link.get('href') for link in links if link.get('href')]
                self.logger.info(f"URLs extraídas com sucesso da tabela 'landcoverfull'. Total: {len(urls)}")
                return urls
            else:
                self.logger.warning("Tabela com ID 'landcoverfull' não encontrada.")
                return []

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro ao acessar a página '{url}': {e}")
            return []
        except Exception as e:
            self.logger.error(f"Erro inesperado durante a extração de URLs: {e}")
            return []
