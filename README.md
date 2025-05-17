# Proyecto_DSII_Francone_Jonathan_Comision_61740
Predicción de Churn: Un Enfoque Avanzado con Ensamble y Análisis de Error

Este proyecto tiene como objetivo desarrollar un sistema robusto para predecir el churn (deserción de clientes) a partir de un dataset desbalanceado, utilizando técnicas avanzadas de machine learning. A lo largo del trabajo se han implementado y comparado varios modelos, entre ellos Regresión Logística, Random Forest y XGBoost. Además, se han aplicado mejoras adicionales (bonus) que incluyen un ensamble basado en stacking y un análisis de error para identificar subgrupos con mayores tasas de error y proponer posibles mejoras.
Contenido
•	Descripción del Proyecto
•	Datos
•	Preprocesamiento
•	Implementación de Modelos
•	Optimización y Evaluación
•	Mejoras Bonus
•	Análisis de Error
•	Conclusiones y Futuras Mejoras
•	Uso y Ejecución
•	Dependencias
Descripción del Proyecto
El objetivo de este trabajo es predecir la deserción de clientes (churn) utilizando un conjunto de datos de clientes. Se desarrollaron distintos modelos de clasificación y se optimizó su desempeño a través de técnicas como:
•	Preprocesamiento Integral: Escalado de variables numéricas y codificación de variables categóricas.
•	Manejo de Datasets Desbalanceados: Uso de SMOTE para equilibrar las clases.
•	Optimización de Hiperparámetros: GridSearchCV con validación cruzada estratificada.
•	Reoptimización del Umbral de Decisión: Ajuste del umbral para maximizar el F1-score.
•	Mejoras Bonus: Implementación de técnicas de ensamble mediante stacking y análisis de error para identificar subgrupos con mayor tasa de error.
Datos
El dataset utilizado contiene 10,000 registros y 12 columnas, entre las que destacan:
•	Variables numéricas: credit_score, age, tenure, balance, products_number, estimated_salary.
•	Variables categóricas: country, gender.
•	Variable objetivo: churn.
La correcta definición y separación de estas variables es fundamental para el éxito en la predicción de churn.
Preprocesamiento
Se aplicó un preprocesamiento que incluye:
•	Escalado: Se utiliza StandardScaler para normalizar las variables numéricas.
•	Codificación One-Hot: Se incorpora OneHotEncoder para transformar las variables categóricas.
•	División de Datos: Se separa el dataset en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%), manteniendo la proporción de clases mediante stratify=y.
•	Balanceo de Clases: Se integra SMOTE para mitigar la desproporción en la variable objetivo.
Implementación de Modelos
Diversos modelos fueron desarrollados utilizando pipelines integrados con preprocesamiento y SMOTE:
1.	Regresión Logística: Modelo base utilizado como referencia.
2.	Random Forest: Modelo basado en árboles, optimizado mediante GridSearchCV, que mostró un alto poder discriminativo.
3.	XGBoost: Modelo de boosting que se entrenó con early stopping para evitar el sobreajuste.
4.	Ensamblado (Stacking Ensemble): Se combina el Random Forest optimizado y una versión de XGBoost (sin early stopping para la integración) mediante un meta-modelo (Regresión Logística).
Optimización y Evaluación
La optimización se realizó mediante:
•	GridSearchCV: Se ajustaron hiperparámetros (n_estimators, max_depth, learning_rate, entre otros) con validación cruzada estratificada utilizando el F1-score como métrica principal.
•	Reoptimización del Umbral: Se evaluaron distintas configuraciones de umbral a partir de la curva Precision-Recall para maximizar la identificación de churn.
•	Métricas Evaluadas: Exactitud, precisión, recall, F1-score, AUC-ROC y Brier Score.
Los modelos basados en árboles (Random Forest y XGBoost) alcanzaron un AUC-ROC de alrededor de 0.85 y F1-scores para la clase churn de aproximadamente 0.61, evidenciando un desempeño robusto.
Mejoras Bonus
Se incorporaron dos mejoras adicionales:
1.	Stacking Ensemble:
o	Se implementó un StackingClassifier que combina los modelos base (Random Forest y XGBoost).
o	El meta-modelo es una Regresión Logística que, sin el parámetro passthrough, recibe únicamente las predicciones numéricas de los modelos base.
2.	Análisis de Error:
•	Se realizó un análisis de los errores de predicción agrupados por grupos de edad y país para identificar subgrupos con mayores tasas de error.
•	Los hallazgos indicaron que clientes de grupos de edad "senior" y "elder" presentan mayores tasas de error, lo que sugiere oportunidades de mejora en la ingeniería de características o preprocesamiento personalizado.
Análisis de Error
El análisis segmentado mostró lo siguiente:
•	Por Grupo de Edad:
o	Los clientes "young" tienen una tasa de error muy baja (≈ 5%).
o	Los grupos "senior" y "elder" presentan tasas de error elevadas (≈ 32.7% y 29.2%, respectivamente).
•	Por País:
•	Se observan diferencias en las tasas de error entre países, indicando que ciertos subgrupos pueden necesitar ajustes adicionales en la estrategia de modelado.
Estos hallazgos guían la mejora focalizada para ajustar el modelo en segmentos específicos.
Conclusiones y Futuras Mejoras
El sistema desarrollado demuestra altos niveles de rendimiento en la predicción de churn, incluso en presencia de datos desbalanceados, con un AUC-ROC cercano a 0.85 y exactitudes entre 81% y 85%. Entre los ajustes realizados, el modelo Random Forest y el XGBoost con early stopping resultaron ser muy sólidos. La combinación en un stacking ensemble aporta robustez y ofrece la posibilidad de mejorar aún más mediante técnicas avanzadas de ensamblado.
Futuras Mejoras:
•	Explorar técnicas adicionales de ensamblado, como stacking avanzado o ensambles basados en stacking con múltiples niveles.
•	Afinar el early stopping adaptativo en XGBoost para mejorar la generalización sin necesidad de eliminarlo en el modelo de ensamble.
•	Profundizar en el análisis de error para diseñar estrategias de ingeniería de características o preprocesamiento específicas para subgrupos con mayores tasas de error (por ejemplo, clientes de mayor edad).
Uso y Ejecución
1.	Instalación de Dependencias:
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib 

2.	Ejecución del Código:
o	El código se puede ejecutar como un cuaderno Jupyter (.ipynb) o como un script Python (.py).
o	Asegúrese de que el dataset (por ejemplo, en formato CSV) esté disponible y correctamente formateado.
3.	Interpretación de Resultados:
•	Se imprimirán métricas de evaluación (clasificación, AUC-ROC, Brier Score) y se generarán gráficos que muestran la tasa de error por grupo demográfico, facilitando la identificación de oportunidades de mejora.
Dependencias
•	Python 3.7+
•	Bibliotecas:
•	pandas
•	numpy
•	scikit-learn
•	imbalanced-learn
•	xgboost
•	matplotlib
Este README resume de manera integral el trabajo realizado, desde la preparación del dataset y el diseño de múltiples modelos, hasta la incorporación de mejoras adicionales mediante el ensamblaje y el análisis de error. El proyecto representa un enfoque avanzado para la predicción de churn, combinando técnicas modernas de machine learning con estrategias de optimización y análisis profundo, y ofrece una sólida base para futuras mejoras y aplicaciones en entornos de retención de clientes.
¡Gracias por interesarte en este trabajo!

