# Funções relacionadas a cálculo e filtro de correlação
import numpy as np
import seaborn as sns
from scipy.stats import stats
import matplotlib.pyplot as plt

from EcoDistrib.common import msg_logger
from EcoDistrib.utils import FileManager, RasterOperations

class CorrelationAnalyzer:
    def __init__(self):
        self.logger = msg_logger

    def calculate_filter_display_heatmap(
            self,
            tiff_paths,
            method="pearson",
            title="Heatmap da Matriz de Correlação",
            save_as=None,
            show=True,
            threshold=0.75,
            new_folder="variaveis_nao_correlacionadas"
        ):
        """
        Calcula a matriz de correlação dos arquivos TIFF, exibe um heatmap e filtra variáveis baseadas no limiar de correlação.

        Parâmetros:
        - tiff_paths (str ou list): Caminho para um diretório ou lista de arquivos TIFF.
        - method (str): Tipo de correlação. Pode ser "pearson", "spearman" ou "kendall".
        - title (str): Título do gráfico de heatmap.
        - save_as (str): Caminho para salvar o gráfico como imagem. Se None, não será salvo.
        - show (bool): Se True, exibe o heatmap.
        - threshold (float): Limite para filtrar variáveis altamente correlacionadas.
        - new_folder (str): Pasta para salvar as variáveis não correlacionadas.

        Logs:
        - Informações e avisos durante o processo.
        """
        try:
            # Calcula a matriz de correlação
            corr_matrix, variables = self.calculate_tiffs_correlation(tiff_paths, method)
            self.logger.info(f"Matriz de correlação calculada com o método {method}.")
            
            # Filtra as variáveis com alta correlação
            corr_matrix, variables = self.filter_correlation(corr_matrix, variables, threshold)
            self.logger.info(f"Variáveis filtradas. Restam {len(variables)} variáveis após aplicar o threshold {threshold}.")

            # Exibe e salva o heatmap
            self.display_correlation_heatmap(corr_matrix, variables, title=title, save_as=save_as, show=show)

            # Copia os arquivos correspondentes às variáveis restantes
            FileManager().copiar_tiffs_variaveis(variables, tiff_paths, new_folder)

        except Exception as e:
            self.logger.error(f"Erro no cálculo do filtro do heatmap: {e}")
            raise

    def calculate_tiffs_correlation(self, tiff_paths, method="pearson"):
        """
        Calcula a matriz de correlação entre múltiplos arquivos TIFF.

        Parâmetros:
        - tiff_paths (str ou list): Caminho para um diretório ou lista de arquivos TIFF.
        - method (str): Método de correlação. Pode ser "pearson", "spearman" ou "kendall".

        Retorno:
        - corr_matrix (np.ndarray): Matriz de correlação entre os arquivos.
        - variables (list): Lista dos nomes das variáveis.

        Exceções:
        - ValueError: Lança erro se o método de correlação for inválido.
        """
        if method not in ["pearson", "spearman", "kendall"]:
            raise ValueError("Método de correlação inválido. Use 'pearson', 'spearman' ou 'kendall'.")

        try:
            # Converte os rasters para matriz
            rasters_array, variables = RasterOperations().raster_to_matrix(tiff_paths)
            self.logger.info("Matriz de dados extraída de rasters.")

            # Substitui NaNs para evitar problemas na correlação
            rasters_array = np.nan_to_num(rasters_array)

            # Calcula a matriz de correlação
            corr_matrix = self.correlation(rasters_array, method=method)
            self.logger.info("Matriz de correlação calculada com sucesso.")

            return corr_matrix, variables

        except Exception as e:
            self.logger.error(f"Erro ao calcular correlação entre TIFFs: {e}")
            raise

    def correlation(self, rasters_array, method="pearson"):
        """
        Calcula a matriz de correlação entre rasters.

        Parâmetros:
        - rasters_array (np.ndarray): Matriz onde cada linha é um raster, cada coluna é um pixel.
        - method (str): Método de correlação ("pearson", "spearman", "kendall").

        Retorno:
        - corr_matrix (np.ndarray): Matriz de correlação.
        """
        try:
            n = len(rasters_array)
            corr_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    # Máscara para evitar NaNs
                    valid_mask = ~np.isnan(rasters_array[i]) & ~np.isnan(rasters_array[j])

                    if np.any(valid_mask):
                        if method == "pearson":
                            corr, _ = stats.pearsonr(rasters_array[i][valid_mask], rasters_array[j][valid_mask])
                        elif method == "spearman":
                            corr, _ = stats.spearmanr(rasters_array[i][valid_mask], rasters_array[j][valid_mask])
                        else:  # "kendall"
                            corr, _ = stats.kendalltau(rasters_array[i][valid_mask], rasters_array[j][valid_mask])

                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                    else:
                        corr_matrix[i, j] = np.nan
                        corr_matrix[j, i] = np.nan

            return corr_matrix

        except Exception as e:
            self.logger.error(f"Erro ao calcular a correlação: {e}")
            raise

    def display_correlation_heatmap(self,corr_matrix, variables, title="Heatmap da Matriz de Correlação", save_as=None, show=True):
        """
        Exibe um heatmap da matriz de correlação e opcionalmente salva a figura como uma imagem.

        Parâmetros:
        - corr_matrix (np.ndarray): Matriz de correlação a ser exibida.
        - variables (list): Lista com os nomes das variáveis para os eixos.
        - title (str): Título do gráfico de heatmap.
        - save_as (str): Caminho para salvar o gráfico como imagem. Se None, o gráfico não será salvo.
        - show (bool): Se True, exibe o heatmap.

        Exceções:
        - ValueError: Lança erro se a matriz ou a lista de variáveis for inválida.
        """
        # Validações
        if corr_matrix.size == 0 or corr_matrix.ndim != 2:
            raise ValueError("A matriz de correlação deve ser uma matriz 2D não vazia.")
        if len(variables) != corr_matrix.shape[0] or len(variables) != corr_matrix.shape[1]:
            raise ValueError("O número de variáveis deve corresponder às dimensões da matriz de correlação.")

        # Configurar o gráfico de heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            xticklabels=variables,
            yticklabels=variables,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Correlação"},
        )
        plt.title(title, fontsize=16)
        plt.xlabel('Variáveis', fontsize=12)
        plt.ylabel('Variáveis', fontsize=12)

        # Exibir o heatmap
        if show:
            plt.show()

        # Salvar o gráfico como imagem, se solicitado
        if save_as:
            try:
                plt.savefig(save_as, format="png", dpi=300, bbox_inches="tight")
                print(f"Heatmap salvo como: {save_as}")
            except Exception as e:
                print(f"Erro ao salvar o heatmap: {e}")

    def filter_correlation(self, corr_matrix, variables, threshold=0.75):
        """
        Remove variáveis altamente correlacionadas de uma matriz de correlação de forma recursiva.

        Parâmetros:
        - corr_matrix (np.ndarray): Matriz de correlação.
        - variables (list): Lista dos nomes das variáveis correspondentes à matriz de correlação.
        - threshold (float): O limite de correlação acima do qual as variáveis serão removidas.

        Retorno:
        - np.ndarray: Matriz de correlação após a remoção de variáveis.
        - list: Lista de variáveis restantes após a remoção.

        Exceções:
        - ValueError: Se a matriz de correlação ou a lista de variáveis for inválida.
        """
        # Validações iniciais
        if corr_matrix.size == 0 or corr_matrix.ndim != 2:
            self.logger.error("A matriz de correlação deve ser uma matriz 2D não vazia.")
            raise ValueError("A matriz de correlação deve ser uma matriz 2D não vazia.")
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            self.logger.error("A matriz de correlação deve ser quadrada.")
            raise ValueError("A matriz de correlação deve ser quadrada.")
        if len(variables) != corr_matrix.shape[0]:
            self.logger.error("O número de variáveis deve corresponder às dimensões da matriz de correlação.")
            raise ValueError("O número de variáveis deve corresponder às dimensões da matriz de correlação.")

        # Loop recursivo para verificar e remover variáveis
        while True:
            # Identifica os índices onde a correlação é maior que o limite (ou menor que o limite negativo)
            indices_corr_altas = np.argwhere(np.triu(corr_matrix, k=1) > threshold)
            indices_corr_baixas = np.argwhere(np.triu(corr_matrix, k=1) < -threshold)

            # Combina os índices
            indices_corr = np.vstack((indices_corr_altas, indices_corr_baixas))

            # Para o loop se não houver pares de correlação acima do threshold
            if len(indices_corr) == 0:
                break

            # Calcula a contagem de correlações altas para cada variável
            contagem_corr = np.sum(corr_matrix > threshold, axis=0) + np.sum(corr_matrix < -threshold, axis=0)

            # Encontra o índice da variável com maior número de correlações acima do threshold
            indice_para_remover = np.argmax(contagem_corr)

            # Log da variável que será removida
            self.logger.info(f"Removendo variável altamente correlacionada: {variables[indice_para_remover]}")

            # Remove a variável da lista de variáveis
            variables.pop(indice_para_remover)

            # Remove a linha e coluna correspondente da matriz de correlação
            corr_matrix = np.delete(corr_matrix, indice_para_remover, axis=0)
            corr_matrix = np.delete(corr_matrix, indice_para_remover, axis=1)

        return corr_matrix, variables
