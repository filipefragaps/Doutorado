"""Módulo 13 — Predição.

Do que o usuário pediu ("falha futura, tipo de falha, probabilidade da falha,
tempo restante até a falha, assinatura futura das harmônicas"), este módulo
implementa o que os dados suportam e documenta explicitamente o que não suportam:

- **tipo de falha**: `fault_classifier.py` — classificação (reusa os modelos
  clássicos do módulo 9 e as redes do módulo 11), com probabilidade por classe.
- **probabilidade da falha**: a distribuição de probabilidade do classificador
  (`predict_proba` / softmax), não uma probabilidade de sobrevivência.
- **assinatura futura das harmônicas**: `forecasting.py` — regressão sequencial
  (LSTM/GRU) treinada para prever o vetor de features N passos à frente, usando
  as janelas do módulo 10.
- **tempo restante até a falha (RUL)**: **não implementado**. Confirmado
  empiricamente: cada rótulo é uma única sessão estacionária de ~4-6h com a
  falha já configurada desde o início, não uma trajetória contínua de
  degradação com um evento de falha registrado. Sem essa trajetória não há
  alvo (tempo-até-falha) para supervisionar um modelo de RUL. Para viabilizar
  isso no futuro, seria necessário registrar um ensaio contínuo em que a falha
  se desenvolve progressivamente ao longo do tempo, com o instante da falha
  (ou de cruzamento de um limiar) registrado.
"""
