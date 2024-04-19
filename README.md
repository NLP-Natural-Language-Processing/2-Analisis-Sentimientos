# 2-Analisis-Sentimientos

1. [¿En qué consiste el Análisis de Sentimientos?](#schema1)
2. [Caso Práctico](#schema2)
3. [Ejercicio: Tweets | NLP Cosine Similarity | TF-IDF](#schema3)

<hr>

<a name="schema1"></a>

## 1. ¿En qué consiste el Análisis de Sentimientos?
- Comprender y analizar la respuesta de las personas descubriendo opiniones, emociones y sentimientos sobre un producto, servicio o entidad (mayoritariamente de redes sociales con alto volumen de respuestas).
- Se basa en el Procesamiento del Lenguaje Natural y la estadística asignando valores al texto (positivo, negativo o neutral). Con ello, identificar el sentimiento global (contento, triste,enfadado,…)

- Alta aplicación para:

  - Identificar la respuesta a mensajes de negocio de las empresas
  - Reajuste de la estrategia de negocio
  - Diseñar una mejor experiencia de cliente
  - Mejorar el producto o servicio
  - Análisis de la percepción de marca
  - Predicción de movimientos en bolsa


<hr>

<a name="schema2"></a>

## 2. Caso Práctico

![Twitter](./img/twitter.png)
Necesitamos:

```python
pip install tweepy
pip install textblob
```

Con la versión básica del API de tweeter da este error `Forbidden: 403 Forbidden 453 - You currently have access to a subset of Twitter API v2 endpoints and limited v1.1 endpoints (e.g. media post, oauth) only. If you need access to this endpoint, you may need a different access level. You can learn more here: https://developer.twitter.com/en/portal/product.` y es por esto:

![Error](./img/error.png)


Voy a usar este dataset: https://www.kaggle.com/datasets/bhavikjikadara/tweets-dataset



<hr>

<a name="schema3"></a>

## 3. Ejercicio: Tweets | NLP Cosine Similarity | TF-IDF