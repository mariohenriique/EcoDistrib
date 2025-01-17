# Funções relacionadas a manipulação de arquivos
import os
import shutil
import zipfile
from EcoDistrib.utils.logger import LoggerManager

class FileManager:
    def __init__(self):
        self.logger = LoggerManager().get_logger()

    def exclude_file(self, output_file):
        """
        Exclui um arquivo e registra uma mensagem indicando o nome e a extensão do arquivo excluído.

        Parâmetros:
        - output_file (str): Caminho do arquivo a ser excluído.

        Exceções tratadas:
        - FileNotFoundError: Arquivo não encontrado.
        - PermissionError: Sem permissão para excluir o arquivo.
        - Outras exceções genéricas serão capturadas e registradas no logger.
        """
        try:
            # Excluir o arquivo
            os.remove(output_file)

            # Obter a extensão do arquivo
            _, extension = os.path.splitext(output_file)

            # Registrar mensagem de confirmação
            self.logger.info(f"Arquivo '{output_file}' com extensão '{extension}' foi excluído com sucesso.")

        except FileNotFoundError:
            # Caso o arquivo não seja encontrado
            self.logger.error(f"Erro: O arquivo '{output_file}' não foi encontrado.")
        except PermissionError:
            # Caso não haja permissão para excluir o arquivo
            self.logger.error(f"Erro: Permissão negada para excluir o arquivo '{output_file}'.")
        except Exception as e:
            # Outros erros inesperados
            self.logger.error(f"Erro ao excluir o arquivo '{output_file}': {e}")

    def extract_exclude_zip_file(self, output_file, output_dir,exclude=True):
        """
        Extrai um arquivo ZIP para o diretório especificado e exclui o arquivo ZIP após a extração.

        Parâmetros:
        - output_file (str): Caminho do arquivo ZIP a ser extraído.
        - output_dir (str): Diretório onde os arquivos extraídos serão armazenados.

        Exceções tratadas:
        - FileNotFoundError: O arquivo ZIP especificado não foi encontrado.
        - zipfile.BadZipFile: O arquivo especificado não é um ZIP válido.
        - Outras exceções genéricas serão capturadas e registradas no logger.
        """
        try:
            # Registrar início do processo de extração
            self.logger.info(f"Iniciando a extração de '{output_file}' para o diretório '{output_dir}'.")

            # Abrir e extrair o arquivo ZIP
            with zipfile.ZipFile(output_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            # Registrar sucesso na extração
            self.logger.info(f"Arquivos extraídos com sucesso para '{output_dir}'.")
            if exclude:
                # Excluir o arquivo ZIP após a extração
                self.exclude_file(output_file)

        except FileNotFoundError:
            self.logger.error(f"Erro: O arquivo ZIP '{output_file}' não foi encontrado.")
        except zipfile.BadZipFile:
            self.logger.error(f"Erro: O arquivo '{output_file}' não é um arquivo ZIP válido.")
        except PermissionError:
            self.logger.error(f"Erro: Permissão negada para acessar ou excluir '{output_file}'.")
        except Exception as e:
            self.logger.error(f"Erro inesperado ao processar o arquivo ZIP '{output_file}': {e}")

    def save_tiff_to_directory(self, tiff_file, tiff_output_dir):
        """
        Move o arquivo TIFF para o diretório especificado e registra uma mensagem indicando a localização final do arquivo.

        Parâmetros:
        - tiff_file (str): Caminho do arquivo TIFF a ser movido.
        - tiff_output_dir (str): Diretório onde o arquivo TIFF será armazenado.

        Exceções:
        - O método registra erros caso ocorram problemas durante a movimentação do arquivo, como falha de permissão ou arquivo inexistente.
        """
        try:
            # Cria o diretório de saída, se não existir
            os.makedirs(tiff_output_dir, exist_ok=True)

            # Define o destino para o arquivo TIFF
            destination = os.path.join(tiff_output_dir, os.path.basename(tiff_file))

            # Move o arquivo para o novo diretório
            os.rename(tiff_file, destination)
            
            # Registra a mensagem de sucesso
            self.logger.info(f"Arquivo TIFF salvo em: {destination}")

        except Exception as e:
            # Registra erro caso ocorra alguma falha
            self.logger.error(f"Erro ao mover o arquivo TIFF '{tiff_file}' para o diretório '{tiff_output_dir}': {e}")

    def move_tif_files(self, destination_dir: str, output_dir: str, extension: str = '.tif') -> None:
        """
        Move arquivos com a extensão especificada (por padrão .tif) do diretório de origem para o diretório de destino
        e remove os diretórios vazios em 'output_dir' após a movimentação.

        Parâmetros:
        - destination_dir (str): Diretório onde os arquivos serão armazenados.
        - output_dir (str): Diretório de origem dos arquivos a serem movidos.
        - extension (str): Extensão dos arquivos a serem movidos. O padrão é '.tif'.

        Exceções:
        - A função pode lançar erros se houver falhas na movimentação dos arquivos ou remoção de pastas vazias.
        """
        try:
            # Cria o diretório de destino, se não existir
            os.makedirs(destination_dir, exist_ok=True)
            self.logger.info(f"Diretório de destino '{destination_dir}' criado ou já existe.")

            # Move arquivos com a extensão especificada
            self._move_files_with_extension(destination_dir, output_dir, extension)

            # Remove diretórios vazios
            self.remove_empty_dirs(output_dir)

        except Exception as e:
            # Registra erro caso ocorra alguma falha
            self.logger.error(f"Erro ao mover os arquivos ou remover pastas vazias: {e}")

    def _move_files_with_extension(self, destination_dir: str, output_dir: str, extension: str) -> None:
        """
        Move todos os arquivos com a extensão especificada do diretório de origem para o diretório de destino.
        Após mover os arquivos, verifica e remove as pastas vazias em 'output_dir'.

        Parâmetros:
        - destination_dir (str): Diretório onde os arquivos serão armazenados.
        - output_dir (str): Diretório de origem dos arquivos.
        - extension (str): Extensão dos arquivos a serem movidos.

        Exceções:
        - Pode lançar erros relacionados à movimentação de arquivos, como FileNotFoundError ou PermissionError.
        """
        try:
            # Mover arquivos com a extensão especificada
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(extension):
                        source_path = os.path.join(root, file)
                        destination_path = os.path.join(destination_dir, file)
                        shutil.move(source_path, destination_path)  # Move o arquivo
                        self.logger.info(f"Arquivo movido para {destination_path}")

        except Exception as e:
            # Registra erro caso ocorra alguma falha ao mover os arquivos
            self.logger.error(f"Erro ao mover arquivos com extensão '{extension}': {e}")

    def remove_empty_dirs(self, output_dir: str) -> None:
        """
        Remove todos os diretórios vazios dentro do diretório 'output_dir', começando pelas subpastas e subindo até a raiz.

        Parâmetros:
        - output_dir (str): Diretório onde as pastas vazias serão removidas.

        Exceções:
        - Pode lançar erros se ocorrerem problemas ao tentar remover pastas, como falha de permissão.
        """
        try:
            # Remover pastas vazias de output_dir até a raiz
            for root, dirs, _ in os.walk(output_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if os.path.exists(dir_path) and not os.listdir(dir_path):  # Verifica se o diretório está vazio
                        os.rmdir(dir_path)
                        self.logger.info(f"Pasta vazia removida: {dir_path}")

                # Verifica se o próprio diretório root está vazio após a remoção das subpastas
                if os.path.exists(root) and not os.listdir(root):
                    os.rmdir(root)
                    self.logger.info(f"Pasta vazia removida: {root}")

        except Exception as e:
            # Registra erro caso ocorra algum problema ao remover pastas
            self.logger.error(f"Erro ao remover diretórios vazios: {e}")

    def listfile(self,tiff_paths):
        """
        Verifica se o caminho fornecido é um diretório ou arquivo e retorna uma lista de caminhos para arquivos TIFF.

        Parâmetros:
        - tiff_paths (str ou list): Caminho para um diretório, um arquivo TIFF, ou uma lista de arquivos.

        Retorno:
        - list: Lista de caminhos para arquivos TIFF válidos.

        Exceções:
        - ValueError: Caso `tiff_paths` não seja um caminho válido ou uma lista de arquivos válidos.
        """
        try:
            # Se `tiff_paths` for um único caminho (string)
            if isinstance(tiff_paths, str):
                if os.path.isdir(tiff_paths):
                    # Lista todos os arquivos com extensão .tif ou .tiff no diretório
                    tiff_paths = [
                        os.path.join(tiff_paths, f) for f in os.listdir(tiff_paths)
                        if f.lower().endswith(('.tif', '.tiff'))
                    ]
                elif os.path.isfile(tiff_paths) and tiff_paths.lower().endswith(('.tif', '.tiff')):
                    # Retorna como lista se for um único arquivo válido
                    tiff_paths = [tiff_paths]
                else:
                    raise ValueError(f"O caminho fornecido '{tiff_paths}' não é um diretório nem um arquivo TIFF válido.")

            # Se `tiff_paths` for uma lista, valida os caminhos
            elif isinstance(tiff_paths, list):
                if not all(isinstance(f, str) and os.path.isfile(f) and f.lower().endswith(('.tif', '.tiff')) for f in tiff_paths):
                    raise ValueError("A lista fornecida contém elementos que não são arquivos TIFF válidos.")
            else:
                raise ValueError("`tiff_paths` deve ser uma string (diretório ou arquivo TIFF) ou uma lista de caminhos.")

            return tiff_paths

        except Exception as e:
            raise ValueError(f"Erro ao processar os caminhos fornecidos: {e}")

    def copiar_tiffs_variaveis(self,variables, tiff_paths, new_folder):
        """
        Copia arquivos TIFF correspondentes às variáveis restantes para uma nova pasta.

        Parâmetros:
        - variables (list): Lista de nomes das variáveis cujos arquivos TIFF devem ser copiados.
        - tiff_paths (str): Caminho para o diretório contendo arquivos TIFF.
        - new_folder (str): Nome da nova pasta onde os arquivos TIFF serão copiados.
        """
        # Criar nova pasta se não existir
        os.makedirs(new_folder, exist_ok=True)

        # Copiar os arquivos TIFF correspondentes às variáveis restantes
        for variable in variables:
            tiff_file_name = f"{variable}.tif"  # Assumindo que a extensão é .tif
            tiff_file_path = os.path.join(tiff_paths, tiff_file_name)

            # Verifica se o arquivo existe antes de copiar
            if os.path.isfile(tiff_file_path):
                shutil.copy2(tiff_file_path, new_folder)
            else:
                print(f"Arquivo não encontrado: {tiff_file_path}")
