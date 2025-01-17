# Avaliação de desempenho dos modelos
class ModelEvaluator:
    def avaliar_modelo(y_true, y_pred, y_prob=None):
        """
        Avalia o desempenho de um modelo preditivo usando métricas comuns.

        Parâmetros:
        y_true (array-like): Valores verdadeiros (1 para presença, 0 para ausência).
        y_pred (array-like): Valores previstos pelo modelo (1 ou 0).
        y_prob (array-like, opcional): Probabilidades previstas para a classe 1.

        Retorna:
        dict: Dicionário com métricas de avaliação.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )

        resultados = {
            "Acurácia": accuracy_score(y_true, y_pred),
            "Precisão": precision_score(y_true, y_pred, zero_division=0),
            "Sensitividade (Recall)": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "Matriz de Confusão": confusion_matrix(y_true, y_pred).tolist(),
        }
        if y_prob is not None:
            resultados["AUC-ROC"] = roc_auc_score(y_true, y_prob)

        return resultados
