# Assignment: Linear Models, Regularization, and Model Selection on Real Data

**Deadline:** Sunday, October 5th, 2025, 23:59

**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

---

### Integrantes del Grupo:

- Hidalgo Eche, Diana
- Llaro Castro, Diego


#### Diferencias entre OLS, Ridge y Lasso

En el modelo de Regresión Lineal por OLS, los coeficientes se estiman únicamente minimizando el error cuadrático medio. Esto genera un ajuste correcto, pero puede verse afectado por problemas de multicolinealidad (alta correlación entre variables) o sobreajuste si el número de predictores es grande.

Con la incorporación de Ridge se observa que los coeficientes se reducen en magnitud conforme aumenta el parámetro de regularización α, aunque ninguno de ellos llega a ser exactamente cero. Esto implica que Ridge distribuye el peso entre todas las variables, controlando la varianza del modelo y mejorando su capacidad de generalización.

Por otro lado, Lasso no solo reduce la magnitud de los coeficientes, sino que algunos de ellos se vuelven exactamente cero para valores suficientemente grandes de α. En consecuencia, Lasso realiza también una selección automática de variables, simplificando el modelo.

Los errores cuadráticos medios fueron los siguientes:
- MSE OLS - forma cerrada: 0.5558915986952438
- MSE GD: 0.5549457340435588
- MSE Ridge: 0.5557665649520098
- MSE Lasso: 0.5544913600832686

Siendo el que tuvo menor error Lasso.

#### Efecto de la tasa de aprendizaje en el descenso de gradiente

El algoritmo de Descenso de Gradiente fue implementado con tres valores de tasa de aprendizaje: 0.01, 0.001 y 0.0001.

Con learning rate = 0.1:
El costo disminuyó de manera rápida en las primeras iteraciones y se alcanzó la convergencia en pocas épocas. Esto permitió aproximarse de forma eficiente a la solución de OLS.

Con learning rate = 0.01:
El descenso fue más estable pero notoriamente más lento. Se requirieron muchas más iteraciones para que la función de costo se acercara al mínimo.

Con learning rate = 0.001:
La convergencia fue extremadamente lenta, al punto que incluso tras varias iteraciones el costo seguía siendo relativamente alto. Esto evidencia que un valor demasiado pequeño hace que el entrenamiento sea ineficiente.


#### Rol de la validación cruzada k-Fold en la selección de la fuerza de regularización

La validación cruzada k-Fold se utilizó para seleccionar el valor óptimo de α en Ridge y Lasso.

El procedimiento consiste en:

Dividir el conjunto de entrenamiento en k pliegues.
Entrenar el modelo con distintos valores de α en k-1 pliegues y evaluar el error en el pliegue restante.
Repetir el proceso k veces y promediar los errores.
De este modo, se comparan los errores promedio para distintos α, eligiendo aquel que minimiza el error de validación.

Los resultados mostraron que:

Para Ridge, el error disminuye al introducir una regularización moderada, y aumenta si α es demasiado grande (por exceso de penalización).
Para Lasso, la validación cruzada identifica un α que equilibra bien la reducción de complejidad con el error de predicción.
