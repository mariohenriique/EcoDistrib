
# EcoDistrib

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A **EcoDistrib** é uma biblioteca Python para modelagem de distribuição de espécies (SDM - Species Distribution Modeling). Ela oferece ferramentas para baixar variáveis ambientais, pré-processar dados (como corte de áreas de interesse, aplicação de matriz de correlação e PCA), e realizar modelagem de distribuição de espécies usando diversos métodos (distância, estatísticos e machine learning). Além disso, a biblioteca calcula métricas de avaliação para os modelos gerados.

## Instalação

Para instalar a biblioteca, utilize o seguinte comando:
Dependências
A biblioteca depende dos seguintes pacotes Python:

- pandas
- numpy
- rasterio
- scikit-learn
- matplotlib
- geopandas
- shapely

Certifique-se de que todas as dependências estão instaladas antes de usar a biblioteca.

## Funcionalidades Principais

1. **Download de Dados Ambientais**  
   Baixa variáveis ambientais de fontes públicas (por exemplo, WorldClim).  
   **Classe:** DataDownloader  
   **Método:** download_data

2. **Manipulação de Shapefiles**  
   Cria shapefiles para países ou estados específicos.  
   **Classe:** ShapefileHandler  
   **Métodos:**  
   - create_shapefile_countries: Cria um shapefile para um ou mais países.  
   - create_shapefile_states: Cria um shapefile para um ou mais estados.

3. **Manipulação de Rasters**  
   Recorta rasters com base em bounding boxes ou polígonos de shapefiles.  
   **Classe:** RasterHandler  
   **Método:** crop_raster

4. **Análise de Correlação**  
   Calcula e exibe matrizes de correlação (Spearman, Kendall, Pearson) entre variáveis ambientais.  
   **Classe:** CorrelationAnalyzer  
   **Métodos:**  
   - calculate_tiffs_correlation: Calcula a matriz de correlação.  
   - display_correlation_heatmap: Exibe um heatmap da matriz de correlação.  
   - calculate_filter_display_heatmap: Filtra variáveis com base na correlação e exibe o heatmap.

5. **Aplicação de PCA**  
   Aplica Análise de Componentes Principais (PCA) nas variáveis ambientais.  
   **Classe:** PCAProcessor  
   **Método:** apply_pca

6. **Preparação de Dados para Modelagem**  
   Gera pseudo-ausências e prepara os dados para modelagem.  
   **Classe:** ModelDataPrepare  
   **Método:** generate_pseudo_absence

7. **Extração de Valores de Rasters**  
   Extrai valores de variáveis ambientais para coordenadas específicas.  
   **Classe:** RasterDataExtract  
   **Método:** get_values

8. **Modelagem de Distribuição de Espécies**  
   Implementa diversos métodos de modelagem:

   - **Métodos de Distância:** Bioclim, Mahalanobis, Euclidiana, Canberra, Chebyshev, Cosseno, Minkowski, Manhattan.  
   - **Métodos Estatísticos:** GLM (Modelo Linear Generalizado), GAM (Modelo Aditivo Generalizado).  
   - **Métodos de Machine Learning:** Random Forest, ANN (Redes Neurais Artificiais), SVM (Máquinas de Vetores de Suporte).  
   - **MaxEnt:** Modelo de entropia máxima.

   **Classes:**  
   - DistanceModeling  
   - StatisticalModeling  
   - MLModeling  
   - MaxentModeling

9. **Avaliação de Modelos**  
   Calcula métricas de avaliação (por exemplo, AUC, TSS, Kappa).  
   **Classe:** ModelEvaluator  
   **Método:** compute_metrics

## Exemplos de Uso

1. **Download de Dados Ambientais**  
```python
from EcoDistrib.utils.data_download import DataDownloader

DataDownloader().download_data(destination_dir='../wordclim', output_dir='')
```

2. **Criação de Shapefiles**  
```python
from EcoDistrib.preprocessing.shapefile_operations import ShapefileHandler

# Cria um shapefile para o Brasil
ShapefileHandler().create_shapefile_countries(["Brazil"], "shapefile/brasil.shp")

# Cria um shapefile para os estados de SP e AC
ShapefileHandler().create_shapefile_states(["SP", "AC"], "shapefile/estados_selecionados.shp")
```

3. **Recorte de Rasters**  
```python
from EcoDistrib.utils import RasterHandler

# Recorta rasters usando um bounding box
RasterHandler().crop_raster(
    raster_path='downloaded_data/wordclim_data',
    output_path='raster_bound',
    method="bounding_box",
    bounding_box=[-46, -20, -42, -16]  # Exemplo de bounding box
)

# Recorta rasters usando um polígono de shapefile
RasterHandler().crop_raster(
    raster_path='downloaded_data/wordclim_data',
    output_path='raster_estados',
    method="polygon",
    shapefile_path="shapefile/estados_selecionados.shp"
)
```

4. **Análise de Correlação**  
```python
from EcoDistrib.preprocessing import CorrelationAnalyzer

# Calcula e exibe a matriz de correlação de Spearman
corr_matrix_spearman, variables = CorrelationAnalyzer().calculate_tiffs_correlation('raster_pais', method='spearman')
CorrelationAnalyzer().display_correlation_heatmap(corr_matrix_spearman, variables, title='Heatmap da Matriz de Correlação Spearman', save_as='correlacao/spearman.png')
```

5. **Aplicação de PCA**  
```python
from EcoDistrib.preprocessing import PCAProcessor

PCAProcessor().apply_pca(input_folder="variaveis_filtradas/", output_folder="raster_pca/", n_components=3)
```

6. **Modelagem de Distribuição de Espécies**  
```python
from EcoDistrib.modeling import DistanceModeling, ModelEvaluator

# Modelagem usando Bioclim
model = DistanceModeling()
map_bioclim = model.sdm_bioclim(df_for_sdm, 'raster_pca', save=True, output_save="sdm/mapa_resultante_bioclim.tif")

# Avaliação do modelo
evaluator = ModelEvaluator(model=model, occurrence_data=df_for_sdm, tiff_paths='raster_pca')
metrics = evaluator.compute_metrics(save=True, output_save='sdm/metrics.csv')
```

## Métricas de Avaliação

- **AUC:** Área sob a curva ROC.
- **TSS:** True Skill Statistic.
- **Kappa:** Coeficiente Kappa.

## Contribuição

Contribuições são bem-vindas! Siga as diretrizes de contribuição no repositório.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.