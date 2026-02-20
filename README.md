# Algoritmos de Clasificación en Machine Learning

## Descripción

Este repositorio contiene material técnico exhaustivo sobre algoritmos fundamentales de clasificación en Machine Learning, desarrollado con rigor académico. El contenido incluye fundamentos matemáticos, formulaciones teóricas completas, análisis de complejidad computacional y aplicaciones prácticas de tres algoritmos esenciales: Regresión Logística, Support Vector Machines (SVM) y K-Nearest Neighbors (KNN).

## Contenido del Repositorio

### Tarjetas Técnicas de Estudio

El archivo `tarjetas_algoritmos_clasificacion.html` proporciona documentación interactiva y comprehensiva que incluye:

- Fundamentos teóricos rigurosos
- Formulaciones matemáticas completas con notación formal
- Derivaciones de optimización y algoritmos de entrenamiento
- Métricas de rendimiento y evaluación
- Análisis de complejidad computacional
- Aplicaciones del mundo real
- Ventajas, limitaciones y consideraciones prácticas

### Cuadro Comparativo

El archivo # Algoritmos de Clasificación en Machine Learning

## Descripción

Este repositorio contiene material técnico exhaustivo sobre algoritmos fundamentales de clasificación en Machine Learning, desarrollado con rigor académico de nivel maestría. El contenido incluye fundamentos matemáticos, formulaciones teóricas completas, análisis de complejidad computacional y aplicaciones prácticas de tres algoritmos esenciales: Regresión Logística, Support Vector Machines (SVM) y K-Nearest Neighbors (KNN).

## Contenido del Repositorio

### Tarjetas Técnicas de Estudio

El archivo `tarjetas_algoritmos_clasificacion.html` proporciona documentación interactiva y comprehensiva que incluye:

- Fundamentos teóricos rigurosos
- Formulaciones matemáticas completas con notación formal
- Derivaciones de optimización y algoritmos de entrenamiento
- Métricas de rendimiento y evaluación
- Análisis de complejidad computacional
- Aplicaciones del mundo real
- Ventajas, limitaciones y consideraciones prácticas

### Cuadro Comparativo

El archivo `cuadro_comparativo_algoritmos_ml.html` ofrece una comparación sistemática de cinco algoritmos de Machine Learning (Árboles de Decisión, Naive Bayes, KNN, SVM, Redes Neuronales), organizada en formato vertical optimizado para exportación a PDF.

## Algoritmos Cubiertos

### 1. Regresión Logística

**Paradigma**: Modelo discriminativo basado en función logística

**Formulación Central**:
```
P(Y=1|x) = σ(β^T x) = 1 / (1 + e^(-β^T x))
```

**Características**:
- Optimización mediante máxima verosimilitud (MLE)
- Métodos de segundo orden: Newton-Raphson, L-BFGS
- Regularización L1 (Lasso) y L2 (Ridge)
- Extensión multinomial vía función softmax
- Interpretabilidad de coeficientes como log-odds ratios

**Complejidad**:
- Entrenamiento: O(n·m·k) con gradiente descendente
- Predicción: O(m) por muestra

**Aplicaciones**:
- Credit scoring y análisis de riesgo crediticio
- Diagnóstico médico y predicción de enfermedades
- Predicción de churn en marketing
- Análisis de propensión de compra

### 2. Support Vector Machines (SVM)

**Paradigma**: Maximización del margen con teoría de aprendizaje estadístico

**Formulación Central (Margen Suave)**:
```
min_(w,b,ξ) (1/2)||w||² + C·Σξᵢ
sujeto a: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

**Características**:
- Formulación dual con multiplicadores de Lagrange
- Kernel trick para clasificación no lineal
- Kernels: Lineal, Polinomial, RBF (Gaussiano), Sigmoide
- Condiciones KKT (Karush-Kuhn-Tucker)
- Solo depende de vectores de soporte (sparse solution)

**Complejidad**:
- Entrenamiento: O(n²·m) a O(n³·m)
- Predicción: O(n_sv·m) donde n_sv es número de vectores de soporte

**Aplicaciones**:
- Reconocimiento facial y clasificación de imágenes
- Clasificación de proteínas en bioinformática
- Reconocimiento de escritura manuscrita (MNIST)
- Detección de intrusiones y anomalías

### 3. K-Nearest Neighbors (KNN)

**Paradigma**: Aprendizaje basado en instancias (lazy learning)

**Formulación Central (Clasificación)**:
```
ŷ = argmax_c Σ_{i∈N_K(x)} 1(yᵢ = c)
```

**Características**:
- No paramétrico y libre de supuestos distribucionales
- Sin fase de entrenamiento explícita
- Métricas de distancia: Euclidiana, Manhattan, Minkowski, Mahalanobis, Coseno
- Estructuras espaciales: KD-Tree, Ball Tree, LSH
- Propiedades teóricas: límite de Cover-Hart, consistencia universal

**Complejidad**:
- Entrenamiento: O(1)
- Predicción naive: O(n·m)
- Predicción con KD-Tree: O(m log n) en baja dimensión

**Aplicaciones**:
- Sistemas de recomendación y filtrado colaborativo
- Imputación de valores faltantes
- Detección de anomalías y outliers
- Reconocimiento de patrones en espacios de baja dimensión

## Fundamentos Matemáticos

El material cubre en profundidad:

### Optimización
- Gradiente descendente y variantes (SGD, mini-batch)
- Métodos de segundo orden (Newton-Raphson, quasi-Newton)
- Optimización convexa y dualidad de Lagrange
- Condiciones KKT para problemas con restricciones

### Teoría de Aprendizaje
- Teorema de Bayes y estimación de máxima verosimilitud
- Bias-variance tradeoff
- Teoría PAC (Probably Approximately Correct)
- Dimensión VC (Vapnik-Chervonenkis)
- Límites de generalización

### Kernels y Espacios de Hilbert
- Kernel trick y mapeos implícitos
- Condiciones de Mercer para kernels válidos
- Reproducing Kernel Hilbert Spaces (RKHS)
- Funciones kernel comunes y sus propiedades

### Métricas y Distancias
- Normas Lp (L1, L2, L∞)
- Distancia de Mahalanobis
- Similaridad coseno y distancia angular
- Propiedades métricas y espacios métricos

## Métricas de Evaluación

El material incluye análisis detallado de:

- **Clasificación Binaria**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Log-Loss
- **Clasificación Multiclase**: Macro/Micro averaging, Confusion Matrix, Cohen's Kappa
- **Evaluación Probabilística**: Calibración de probabilidades, Brier Score, Log-Loss
- **Validación**: K-Fold Cross-Validation, Stratified CV, Leave-One-Out CV
- **Clustering**: Silhouette Score, Calinski-Harabasz Index

## Análisis de Complejidad

Cada algoritmo incluye análisis exhaustivo de:

### Complejidad Temporal
- Big-O notation para entrenamiento y predicción
- Análisis de caso promedio y peor caso
- Escalabilidad con número de muestras (n) y características (m)

### Complejidad Espacial
- Requisitos de memoria durante entrenamiento
- Almacenamiento del modelo entrenado
- Trade-offs entre tiempo y espacio

### Consideraciones Prácticas
- Escalabilidad a Big Data
- Paralelización y computación distribuida
- Aceleración por hardware (GPU/TPU)

## Estructura de Archivos

```
.
├── README.md                                  # Este archivo
├── tarjetas_algoritmos_clasificacion.html    # Material de estudio principal
└── cuadro_comparativo_algoritmos_ml.html     # Cuadro comparativo en formato tabla
```

## Uso del Material

### Visualización de Tarjetas Técnicas

1. Abrir `tarjetas_algoritmos_clasificacion.html` en cualquier navegador web moderno
2. El renderizado de ecuaciones matemáticas utiliza MathJax (requiere conexión a internet)
3. Las tarjetas están optimizadas para lectura en pantalla y exportación a PDF

### Exportación a PDF

Para crear versiones PDF del material:

**Opción 1: Desde el navegador**
- Abrir el archivo HTML
- Usar función "Imprimir" (Ctrl+P / Cmd+P)
- Seleccionar "Guardar como PDF"
- Configurar márgenes y orientación según preferencia

**Opción 2: Herramientas de línea de comandos**
```bash
# Usando wkhtmltopdf
wkhtmltopdf --enable-javascript tarjetas_algoritmos_clasificacion.html output.pdf

# Usando Chrome headless
chrome --headless --print-to-pdf=output.pdf tarjetas_algoritmos_clasificacion.html
```

## Requisitos Técnicos

### Para Visualización
- Navegador web moderno (Chrome, Firefox, Safari, Edge)
- JavaScript habilitado
- Conexión a internet (para cargar MathJax CDN)

### Para Desarrollo/Modificación
- Editor de texto o IDE
- Conocimiento de HTML5 y CSS3
- Familiaridad con LaTeX/MathJax para edición de ecuaciones

## Audiencia Objetivo

Este material está diseñado para:

- Estudiantes de maestría en Computer Science, Data Science o campos relacionados
- Profesionales de Machine Learning buscando profundizar fundamentos teóricos
- Investigadores que requieren comprensión matemática rigurosa
- Educadores que preparan cursos avanzados de ML

## Prerrequisitos Recomendados

Para aprovechar completamente el material, se recomienda familiaridad con:

### Matemáticas
- Álgebra lineal: vectores, matrices, espacios vectoriales, autovalores
- Cálculo multivariable: derivadas parciales, gradientes, Hessiana
- Probabilidad y estadística: variables aleatorias, distribuciones, teorema de Bayes
- Optimización: programación convexa, métodos numéricos

### Machine Learning
- Conceptos básicos de aprendizaje supervisado
- Overfitting, underfitting, regularización
- Validación cruzada y evaluación de modelos
- Preprocesamiento de datos (normalización, encoding)

### Programación (Opcional)
- Python con librerías: scikit-learn, NumPy, pandas
- Comprensión de complejidad algorítmica
- Experiencia práctica implementando algoritmos

## Referencias Bibliográficas

### Textos Fundamentales

1. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

2. **Bishop, C. M. (2006)**. *Pattern Recognition and Machine Learning*. Springer.

3. **Murphy, K. P. (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press.

4. **Vapnik, V. N. (1995)**. *The Nature of Statistical Learning Theory*. Springer-Verlag.

5. **Duda, R. O., Hart, P. E., & Stork, D. G. (2001)**. *Pattern Classification* (2nd ed.). Wiley-Interscience.

### Artículos Seminales

1. **Cover, T., & Hart, P. (1967)**. Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.

2. **Cortes, C., & Vapnik, V. (1995)**. Support-vector networks. *Machine Learning*, 20(3), 273-297.

3. **Platt, J. (1998)**. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. *Microsoft Research Technical Report MSR-TR-98-14*.

4. **Cox, D. R. (1958)**. The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B*, 20(2), 215-232.

### Recursos Online

- **Stanford CS229**: Machine Learning - http://cs229.stanford.edu/
- **MIT 6.867**: Machine Learning - https://ocw.mit.edu/
- **scikit-learn Documentation**: https://scikit-learn.org/stable/documentation.html
- **Kernel Methods Tutorial**: http://www.kernel-machines.org/

## Temas Avanzados Relacionados

### Extensiones de los Algoritmos

**Regresión Logística**:
- Elastic Net (combinación L1 + L2)
- Sparse logistic regression
- Bayesian logistic regression
- Polytomous regression (extensiones ordinales)

**SVM**:
- ν-SVM (parametrización alternativa)
- One-class SVM para detección de anomalías
- Support Vector Regression (SVR)
- Multiple Kernel Learning (MKL)
- Least Squares SVM (LS-SVM)

**KNN**:
- Large Margin Nearest Neighbor (LMNN)
- Neighborhood Components Analysis (NCA)
- Local Outlier Factor (LOF)
- Approximate Nearest Neighbors (ANN)

### Campos de Investigación Activa

- Deep metric learning para aprendizaje de representaciones
- Neural networks con kernel methods
- Fairness en algoritmos de clasificación
- Interpretabilidad y explicabilidad (XAI)
- Online learning y adaptive algorithms
- Quantum machine learning

## Contribuciones

Este material fue desarrollado con fines educativos. Para sugerencias, correcciones o extensiones del contenido, se aceptan contribuciones siguiendo las siguientes pautas:

1. Mantener rigor matemático y precisión formal
2. Incluir referencias bibliográficas apropiadas
3. Verificar notación y coherencia con el resto del material
4. Asegurar que las ecuaciones renderizen correctamente con MathJax

## Notas de Implementación

### Consideraciones Prácticas

**Regresión Logística**:
- Siempre verificar convergencia monitoreando log-likelihood
- Usar regularización para evitar separación perfecta
- Considerar feature engineering (interacciones, transformaciones)
- Calibrar probabilidades si se usan para toma de decisiones

**SVM**:
- Normalización de características es crítica
- Grid search con CV para C y γ en kernel RBF
- Considerar LinearSVC (LIBLINEAR) para datasets grandes con kernel lineal
- Verificar número de vectores de soporte (indicador de complejidad)

**KNN**:
- Normalización OBLIGATORIA antes de calcular distancias
- Experimentar con diferentes métricas según naturaleza de datos
- Usar CV para seleccionar K óptimo
- Considerar algoritmos aproximados (FAISS, Annoy) para escala

### Librerías Recomendadas

**Python**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
```

**R**:
```r
library(glmnet)        # Regresión logística con regularización
library(e1071)         # SVM
library(class)         # KNN
library(caret)         # Framework de ML
```

## Licencia

Este material es de naturaleza educativa y académica. Se distribuye con el propósito de facilitar el aprendizaje y la comprensión profunda de algoritmos de Machine Learning.

## Contacto y Soporte

Para preguntas académicas, discusiones técnicas o reportar errores en el material, por favor abrir un issue en este repositorio.

---

**Nota**: Este material asume familiaridad con matemáticas avanzadas y conceptos de Machine Learning. Se recomienda complementar con implementaciones prácticas y experimentación en datasets reales para consolidar el conocimiento teórico.

**Última actualización**: Febrero 2026 ofrece una comparación sistemática de cinco algoritmos de Machine Learning (Árboles de Decisión, Naive Bayes, KNN, SVM, Redes Neuronales), organizada en formato vertical optimizado para exportación a PDF.

## Algoritmos Cubiertos

### 1. Regresión Logística

**Paradigma**: Modelo discriminativo basado en función logística

**Formulación Central**:
```
P(Y=1|x) = σ(β^T x) = 1 / (1 + e^(-β^T x))
```

**Características**:
- Optimización mediante máxima verosimilitud (MLE)
- Métodos de segundo orden: Newton-Raphson, L-BFGS
- Regularización L1 (Lasso) y L2 (Ridge)
- Extensión multinomial vía función softmax
- Interpretabilidad de coeficientes como log-odds ratios

**Complejidad**:
- Entrenamiento: O(n·m·k) con gradiente descendente
- Predicción: O(m) por muestra

**Aplicaciones**:
- Credit scoring y análisis de riesgo crediticio
- Diagnóstico médico y predicción de enfermedades
- Predicción de churn en marketing
- Análisis de propensión de compra

### 2. Support Vector Machines (SVM)

**Paradigma**: Maximización del margen con teoría de aprendizaje estadístico

**Formulación Central (Margen Suave)**:
```
min_(w,b,ξ) (1/2)||w||² + C·Σξᵢ
sujeto a: yᵢ(w^T xᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

**Características**:
- Formulación dual con multiplicadores de Lagrange
- Kernel trick para clasificación no lineal
- Kernels: Lineal, Polinomial, RBF (Gaussiano), Sigmoide
- Condiciones KKT (Karush-Kuhn-Tucker)
- Solo depende de vectores de soporte (sparse solution)

**Complejidad**:
- Entrenamiento: O(n²·m) a O(n³·m)
- Predicción: O(n_sv·m) donde n_sv es número de vectores de soporte

**Aplicaciones**:
- Reconocimiento facial y clasificación de imágenes
- Clasificación de proteínas en bioinformática
- Reconocimiento de escritura manuscrita (MNIST)
- Detección de intrusiones y anomalías

### 3. K-Nearest Neighbors (KNN)

**Paradigma**: Aprendizaje basado en instancias (lazy learning)

**Formulación Central (Clasificación)**:
```
ŷ = argmax_c Σ_{i∈N_K(x)} 1(yᵢ = c)
```

**Características**:
- No paramétrico y libre de supuestos distribucionales
- Sin fase de entrenamiento explícita
- Métricas de distancia: Euclidiana, Manhattan, Minkowski, Mahalanobis, Coseno
- Estructuras espaciales: KD-Tree, Ball Tree, LSH
- Propiedades teóricas: límite de Cover-Hart, consistencia universal

**Complejidad**:
- Entrenamiento: O(1)
- Predicción naive: O(n·m)
- Predicción con KD-Tree: O(m log n) en baja dimensión

**Aplicaciones**:
- Sistemas de recomendación y filtrado colaborativo
- Imputación de valores faltantes
- Detección de anomalías y outliers
- Reconocimiento de patrones en espacios de baja dimensión

## Fundamentos Matemáticos

El material cubre en profundidad:

### Optimización
- Gradiente descendente y variantes (SGD, mini-batch)
- Métodos de segundo orden (Newton-Raphson, quasi-Newton)
- Optimización convexa y dualidad de Lagrange
- Condiciones KKT para problemas con restricciones

### Teoría de Aprendizaje
- Teorema de Bayes y estimación de máxima verosimilitud
- Bias-variance tradeoff
- Teoría PAC (Probably Approximately Correct)
- Dimensión VC (Vapnik-Chervonenkis)
- Límites de generalización

### Kernels y Espacios de Hilbert
- Kernel trick y mapeos implícitos
- Condiciones de Mercer para kernels válidos
- Reproducing Kernel Hilbert Spaces (RKHS)
- Funciones kernel comunes y sus propiedades

### Métricas y Distancias
- Normas Lp (L1, L2, L∞)
- Distancia de Mahalanobis
- Similaridad coseno y distancia angular
- Propiedades métricas y espacios métricos

## Métricas de Evaluación

El material incluye análisis detallado de:

- **Clasificación Binaria**: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Log-Loss
- **Clasificación Multiclase**: Macro/Micro averaging, Confusion Matrix, Cohen's Kappa
- **Evaluación Probabilística**: Calibración de probabilidades, Brier Score, Log-Loss
- **Validación**: K-Fold Cross-Validation, Stratified CV, Leave-One-Out CV
- **Clustering**: Silhouette Score, Calinski-Harabasz Index

## Análisis de Complejidad

Cada algoritmo incluye análisis exhaustivo de:

### Complejidad Temporal
- Big-O notation para entrenamiento y predicción
- Análisis de caso promedio y peor caso
- Escalabilidad con número de muestras (n) y características (m)

### Complejidad Espacial
- Requisitos de memoria durante entrenamiento
- Almacenamiento del modelo entrenado
- Trade-offs entre tiempo y espacio

### Consideraciones Prácticas
- Escalabilidad a Big Data
- Paralelización y computación distribuida
- Aceleración por hardware (GPU/TPU)

## Estructura de Archivos

```
.
├── README.md                                  # Este archivo
├── tarjetas_algoritmos_clasificacion.html    # Material de estudio principal
└── cuadro_comparativo_algoritmos_ml.html     # Cuadro comparativo en formato tabla
```

## Uso del Material

### Visualización de Tarjetas Técnicas

1. Abrir `tarjetas_algoritmos_clasificacion.html` en cualquier navegador web moderno
2. El renderizado de ecuaciones matemáticas utiliza MathJax (requiere conexión a internet)
3. Las tarjetas están optimizadas para lectura en pantalla y exportación a PDF

### Exportación a PDF

Para crear versiones PDF del material:

**Opción 1: Desde el navegador**
- Abrir el archivo HTML
- Usar función "Imprimir" (Ctrl+P / Cmd+P)
- Seleccionar "Guardar como PDF"
- Configurar márgenes y orientación según preferencia

**Opción 2: Herramientas de línea de comandos**
```bash
# Usando wkhtmltopdf
wkhtmltopdf --enable-javascript tarjetas_algoritmos_clasificacion.html output.pdf

# Usando Chrome headless
chrome --headless --print-to-pdf=output.pdf tarjetas_algoritmos_clasificacion.html
```

## Requisitos Técnicos

### Para Visualización
- Navegador web moderno (Chrome, Firefox, Safari, Edge)
- JavaScript habilitado
- Conexión a internet (para cargar MathJax CDN)

### Para Desarrollo/Modificación
- Editor de texto o IDE
- Conocimiento de HTML5 y CSS3
- Familiaridad con LaTeX/MathJax para edición de ecuaciones

## Audiencia Objetivo

Este material está diseñado para:

- Estudiantes de maestría en Computer Science, Data Science o campos relacionados
- Profesionales de Machine Learning buscando profundizar fundamentos teóricos
- Investigadores que requieren comprensión matemática rigurosa
- Educadores que preparan cursos avanzados de ML

## Prerrequisitos Recomendados

Para aprovechar completamente el material, se recomienda familiaridad con:

### Matemáticas
- Álgebra lineal: vectores, matrices, espacios vectoriales, autovalores
- Cálculo multivariable: derivadas parciales, gradientes, Hessiana
- Probabilidad y estadística: variables aleatorias, distribuciones, teorema de Bayes
- Optimización: programación convexa, métodos numéricos

### Machine Learning
- Conceptos básicos de aprendizaje supervisado
- Overfitting, underfitting, regularización
- Validación cruzada y evaluación de modelos
- Preprocesamiento de datos (normalización, encoding)

### Programación (Opcional)
- Python con librerías: scikit-learn, NumPy, pandas
- Comprensión de complejidad algorítmica
- Experiencia práctica implementando algoritmos

## Referencias Bibliográficas

### Textos Fundamentales

1. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

2. **Bishop, C. M. (2006)**. *Pattern Recognition and Machine Learning*. Springer.

3. **Murphy, K. P. (2012)**. *Machine Learning: A Probabilistic Perspective*. MIT Press.

4. **Vapnik, V. N. (1995)**. *The Nature of Statistical Learning Theory*. Springer-Verlag.

5. **Duda, R. O., Hart, P. E., & Stork, D. G. (2001)**. *Pattern Classification* (2nd ed.). Wiley-Interscience.

### Artículos Seminales

1. **Cover, T., & Hart, P. (1967)**. Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.

2. **Cortes, C., & Vapnik, V. (1995)**. Support-vector networks. *Machine Learning*, 20(3), 273-297.

3. **Platt, J. (1998)**. Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. *Microsoft Research Technical Report MSR-TR-98-14*.

4. **Cox, D. R. (1958)**. The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B*, 20(2), 215-232.

### Recursos Online

- **Stanford CS229**: Machine Learning - http://cs229.stanford.edu/
- **MIT 6.867**: Machine Learning - https://ocw.mit.edu/
- **scikit-learn Documentation**: https://scikit-learn.org/stable/documentation.html
- **Kernel Methods Tutorial**: http://www.kernel-machines.org/

## Temas Avanzados Relacionados

### Extensiones de los Algoritmos

**Regresión Logística**:
- Elastic Net (combinación L1 + L2)
- Sparse logistic regression
- Bayesian logistic regression
- Polytomous regression (extensiones ordinales)

**SVM**:
- ν-SVM (parametrización alternativa)
- One-class SVM para detección de anomalías
- Support Vector Regression (SVR)
- Multiple Kernel Learning (MKL)
- Least Squares SVM (LS-SVM)

**KNN**:
- Large Margin Nearest Neighbor (LMNN)
- Neighborhood Components Analysis (NCA)
- Local Outlier Factor (LOF)
- Approximate Nearest Neighbors (ANN)

### Campos de Investigación Activa

- Deep metric learning para aprendizaje de representaciones
- Neural networks con kernel methods
- Fairness en algoritmos de clasificación
- Interpretabilidad y explicabilidad (XAI)
- Online learning y adaptive algorithms
- Quantum machine learning

## Contribuciones

Este material fue desarrollado con fines educativos. Para sugerencias, correcciones o extensiones del contenido, se aceptan contribuciones siguiendo las siguientes pautas:

1. Mantener rigor matemático y precisión formal
2. Incluir referencias bibliográficas apropiadas
3. Verificar notación y coherencia con el resto del material
4. Asegurar que las ecuaciones renderizen correctamente con MathJax

## Notas de Implementación

### Consideraciones Prácticas

**Regresión Logística**:
- Siempre verificar convergencia monitoreando log-likelihood
- Usar regularización para evitar separación perfecta
- Considerar feature engineering (interacciones, transformaciones)
- Calibrar probabilidades si se usan para toma de decisiones

**SVM**:
- Normalización de características es crítica
- Grid search con CV para C y γ en kernel RBF
- Considerar LinearSVC (LIBLINEAR) para datasets grandes con kernel lineal
- Verificar número de vectores de soporte (indicador de complejidad)

**KNN**:
- Normalización OBLIGATORIA antes de calcular distancias
- Experimentar con diferentes métricas según naturaleza de datos
- Usar CV para seleccionar K óptimo
- Considerar algoritmos aproximados (FAISS, Annoy) para escala

### Librerías Recomendadas

**Python**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
```

**R**:
```r
library(glmnet)        # Regresión logística con regularización
library(e1071)         # SVM
library(class)         # KNN
library(caret)         # Framework de ML
```

## Licencia

Este material es de naturaleza educativa y académica. Se distribuye con el propósito de facilitar el aprendizaje y la comprensión profunda de algoritmos de Machine Learning.

## Contacto y Soporte

Para preguntas académicas, discusiones técnicas o reportar errores en el material, por favor abrir un issue en este repositorio.

---

**Nota**: Este material asume familiaridad con matemáticas avanzadas y conceptos de Machine Learning. Se recomienda complementar con implementaciones prácticas y experimentación en datasets reales para consolidar el conocimiento teórico.

**Última actualización**: Febrero 2026
