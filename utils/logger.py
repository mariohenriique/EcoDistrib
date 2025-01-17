import logging

class LoggerManager:
    def __init__(self, log_file="sdm.log", log_level=logging.INFO):
        """
        Configura o logger padrão.
        
        :param log_file: Nome do arquivo onde os logs serão salvos.
        :param log_level: Nível de severidade dos logs (default: INFO).
        """
        self.logger = logging.getLogger("project_logger")
        self.logger.setLevel(log_level)
        
        # Evita adicionar múltiplos handlers ao logger
        if not self.logger.handlers:
            # Formato do log
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            
            # Handler para log em arquivo
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Handler opcional para log no console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Retorna o logger configurado.
        """
        return self.logger
