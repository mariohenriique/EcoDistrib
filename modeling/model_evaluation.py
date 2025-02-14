import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from EcoDistrib.outputs import MapGenerator
from EcoDistrib.modeling import ModelDataPrepare
from EcoDistrib.preprocessing import RasterDataExtract
from EcoDistrib.common import msg_logger

class ModelEvaluator:
    def __init__(self, model, occurrence_data, tiff_paths, lat_col='decimalLatitude', lon_col='decimalLongitude',profile=None,formato='GTiff'):
        self.model = model  # Instância de DistanceModeling
        self.model_name = self._get_model_name()  # Novo método para extrair o nome

        if profile is None:
            _,self.profile = MapGenerator().create_synthetic_raster(formato=formato)
        else:
            self.profile = profile
        self.occurrence_data = occurrence_data  # DataFrame com presenças
        self.tiff_paths = tiff_paths  # Caminho para os rasters
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.presence_scores = None
        self.background_scores = None
        self.logger = msg_logger

    def _get_model_name(self):
        """Extrai o nome do modelo baseado na classe ou atributo específico."""
        if hasattr(self.model, 'model_type'):  # Se o modelo tiver um atributo de nome
            return self.model.model_type
        else:  # Fallback: nome da classe
            return self.model.__class__.__name__

    def _get_presence_scores(self):
        # Usa get_values do seu código existente para extrair valores das presenças
        presence_coords = self.occurrence_data[[self.lon_col, self.lat_col]].values.tolist()
        self.presence_scores = np.array([
            val[0] for val in RasterDataExtract().get_values(
                raster_path=self.tiff_paths,
                coordinates=presence_coords
            )
        ])

    def _get_background_scores(self, n_background=1000):
        # Gera pseudo-ausências usando seu método existente
        pseudo_absences = ModelDataPrepare().generate_pseudo_absence(
            occurrence_data=self.occurrence_data,
            tiff_paths=self.tiff_paths,
            n_pseudo_ausencias=n_background
        )
        # Extrai valores das pseudo-ausências
        background_coords = pseudo_absences[[self.lon_col, self.lat_col]].values.tolist()
        self.background_scores = np.array([
            val[0] for val in RasterDataExtract().get_values(
                raster_path=self.tiff_paths,
                coordinates=background_coords
            )
        ])

    def compute_metrics(self, n_background=None, threshold=0.5,save=False,output_save=''):
        # Define n_background como o número de presenças por padrão
        if n_background is None:
            n_background = min(len(self.occurrence_data), 10000)

        self._get_presence_scores()
        self._get_background_scores(n_background)
        
        y_true = np.concatenate([np.ones_like(self.presence_scores), np.zeros_like(self.background_scores)])
        y_scores = np.concatenate([self.presence_scores, self.background_scores])
        
        auc_roc = roc_auc_score(y_true, y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'model':self.model_name,
            'auc_roc': auc_roc,
            'acuracia': accuracy_score(y_true, y_pred),
            'precisao': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'tss': (tp / (tp + fn)) - (fp / (fp + tn)),
            'vp': tp, 'vn': tn, 'fp': fp, 'fn': fn
        }

        if save:
            self.save_metrics(
                metrics_dict=metrics,
                output_path=output_save,
                model_name=self.model_name,  # Nome dinâmico do modelo
            )

        return metrics

    def save_metrics(self, metrics_dict, output_path, model_name):
        """
        Salva métricas em arquivo CSV, substituindo entradas existentes do mesmo modelo.
        
        Parâmetros:
        - metrics_dict (dict): Dicionário com métricas a serem salvas.
        - output_path (str): Caminho do arquivo CSV.
        - model_name (str): Nome do modelo (ex: 'Bioclim', 'RF', 'ANN').
        """
        try:
            import pandas as pd
            from datetime import datetime

            # Adiciona metadados
            metrics_dict['model'] = model_name
            metrics_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Cria DataFrame com os novos dados
            new_df = pd.DataFrame([metrics_dict])

            # Verifica se o arquivo já existe
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)
                
                # Remove entradas do mesmo modelo (se existirem)
                existing_df = existing_df[existing_df['model'] != model_name]
                
                # Combina com os novos dados
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df

            # Salva o arquivo atualizado
            final_df.to_csv(output_path, index=False)

            self.logger.info(f"Métricas salvas/atualizadas em: {output_path}")

        except Exception as e:
            self.logger.error(f"Falha ao salvar métricas: {e}")
            raise