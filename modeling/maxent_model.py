# Funções específicas para Maxent
import os
import subprocess

from EcoDistrib.common import msg_logger
from EcoDistrib.utils import RasterConverter

class MaxentModeling:
    def __init__(self):
        self.logger = msg_logger

    def sdm_maxent(
            self,
            temp_csv,
            camada_ambiental_dir,
            output_dir="maxent",
            maxent_jar_path="maxent.jar",
            nodata_value=-9999,
            projection_layers_dir=None,
            output_format="logistic",
            maximum_iterations=500,
            regularization_multiplier=1.0,
            replicates=1,
            replicate_type="crossvalidate",
            random_seed=False,
            responsecurves=False,
            jackknife=False,
            threads=1
        ):
        """
        Executa o Maxent no modo headless, configurando os diretórios de entrada, saída e parâmetros, 
        e captura a saída de execução em um log.

        Parâmetros:
        - temp_csv (str): 
            Caminho para o arquivo CSV com as ocorrências.
        - camada_ambiental_dir (str): 
            Diretório contendo os arquivos das camadas ambientais.
        - output_dir (str): 
            Diretório para salvar os resultados do Maxent.
        - maxent_jar_path (str): 
            Caminho para o arquivo .jar do Maxent.
        - nodata_value (float ou int): 
            Valor a ser usado como 'no data' (padrão é -9999).
        - projection_layers_dir (str ou None): 
            Diretório para projeções futuras ou climáticas (opcional).
        - output_format (str): 
            Formato dos resultados ('logistic', 'raw', 'cloglog').
        - maximum_iterations (int): 
            Número máximo de iterações para o treinamento.
        - regularization_multiplier (float): 
            Multiplicador de regularização para evitar overfitting.
        - replicates (int): 
            Número de replicações do modelo.
        - replicate_type (str): 
            Tipo de replicação ('crossvalidate', 'bootstrap', 'subsample').
        - random_seed (bool): 
            Se True, usa uma semente aleatória diferente para cada execução.
        - responsecurves (bool): 
            Se True, gera curvas de resposta.
        - jackknife (bool): 
            Se True, realiza análise de jackknife.
        - threads (int): 
            Número de threads a serem usadas para execução paralela.
        """

        # Verifica se há arquivos .asc no diretório das camadas ambientais
        has_asc_files = any(file.endswith(".asc") for file in os.listdir(camada_ambiental_dir))

        if not has_asc_files:
            self.logger.info(f"Arquivos .asc não encontrados no diretório {camada_ambiental_dir}. Iniciando conversão de .tif para .asc.")
            RasterConverter().convert_tif_to_asc(input_dir=camada_ambiental_dir, output_dir=camada_ambiental_dir, nodata_value=nodata_value)

        # Criar diretório de saída, se não existir
        os.makedirs(output_dir, exist_ok=True)

        # Caminho do arquivo de log para depuração
        log_path = os.path.join(output_dir, "maxent_log.txt")

        # Configurar comando para executar o Maxent
        command = [
            "java", "-mx1024m", "-jar", maxent_jar_path,
            f"environmentallayers={camada_ambiental_dir}",
            f"samplesfile={temp_csv}",
            f"outputdirectory={output_dir}",
            f"outputformat={output_format}",
            f"nodata={nodata_value}",
            f"maximumiterations={maximum_iterations}",
            f"betamultiplier={regularization_multiplier}",
            f"replicates={replicates}",
            f"replicatetype={replicate_type}",
            f"randomseed={'true' if random_seed else 'false'}",
            f"responsecurves={'true' if responsecurves else 'false'}",
            f"jackknife={'true' if jackknife else 'false'}",
            f"threads={threads}",
            "autorun",
            "redoifexists"
        ]

        # Adicionar camadas de projeção, se especificadas
        if projection_layers_dir:
            command.append(f"projectionlayers={projection_layers_dir}")

        # Executar comando e capturar saída no log
        try:
            with open(log_path, "w") as log_file:
                result = subprocess.run(command, stdout=log_file, stderr=log_file, text=True)

            # Checar status de execução
            if result.returncode == 0:
                self.logger.info(f"Execução do Maxent concluída com sucesso. Resultados salvos em: {output_dir}")
            else:
                self.logger.error("Erro na execução do Maxent.")
                self.logger.error(f"Verifique o arquivo de log para mais detalhes: {log_path}")
        except Exception as e:
            self.logger.error(f"Erro ao executar o Maxent: {str(e)}")
