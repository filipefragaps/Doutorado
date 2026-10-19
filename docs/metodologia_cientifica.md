# Abordagem espectral com assinaturas harmônicas e interharmônicas para desagregação de cargas e manutenção preditiva em sistemas trifásicos

**Resumo.** Este trabalho apresenta uma metodologia de análise espectral que
utiliza diretamente componentes harmônicas e interharmônicas, fornecidas por
analisadores de energia, para realizar simultaneamente a detecção de falhas
iminentes e a desagregação de cargas em sistemas trifásicos. Os dados são
obtidos em bancadas experimentais com retificador trifásico alimentando carga
em corrente contínua e um conjunto moto-máquina ajustável submetido a
desequilíbrios de massa, desalinhamentos e defeitos em rolamentos. O banco de
dados contempla condições normais de operação, falhas isoladas, elétricas ou
mecânicas e falhas combinadas, de modo a registrar assinaturas espectrais
representativas de cada cenário. As amplitudes das componentes espectrais nas
três fases constituem vetores de características que alimentam modelos
dedicados: um voltado ao diagnóstico de modos de falha (rede neural
recorrente) e outro à desagregação de cargas em operação simultânea. Os
resultados confirmam a viabilidade de um sistema não invasivo e em tempo real,
capaz de identificar cargas ativas e detectar precocemente anomalias,
fortalecendo estratégias de manutenção preditiva em ambientes industriais.

Este documento estrutura a metodologia da pesquisa de forma pronta para
adaptação direta ao capítulo de metodologia da tese e à seção de métodos de um
artigo. Toda referência a código aponta para o pacote `harmonic_signatures`
(`src/harmonic_signatures/`), de modo que os resultados sejam reproduzíveis a
partir deste repositório.

## Nota de aderência entre o resumo da tese e o que está implementado

Esta seção existe para deixar explícito, sem ambiguidade, onde a implementação
atual bate com o resumo acima e onde ela diverge — evitando que uma leitura
parcial dos relatórios anteriores (focados na sensibilidade mecânica isolada)
seja confundida com o objetivo central do trabalho.

- **"RET N PARAF" = parafusos retirados, confirmado diretamente com o
  pesquisador responsável pelo ensaio.** A bancada moto-máquina permite
  desbalancear removendo parafusos de um disco calibrado; "RET 6/14/15 PARAF"
  é literalmente a quantidade de parafusos retirados para cada grau de
  desbalanceamento (RET 15 PARAF combina isso com desalinhamento). Isso
  confirma a categorização "mecânica pura" (Seção 2) — os rótulos não têm
  nenhuma relação com o retificador — e explica achados anteriores de
  confusão entre graus próximos de desbalanceamento (ex.: `S/ PARAF` vs.
  `RET 14 PARAF` na matriz de confusão, Seção 5 dos relatórios visuais): são
  graus quantitativos do mesmo tipo de falha, não condições qualitativamente
  distintas, então proximidade na assinatura espectral é esperada, não uma
  anomalia.
- **Defeitos em rolamentos**: citados no resumo como parte do desenho
  experimental da bancada, mas **não estão presentes no `BANCO_DADOS_
  CONSOLIDADO.xlsx` atual** — os 17 rótulos disponíveis cobrem desbalanceamento,
  desalinhamento, falha elétrica (diodo/retificador) e combinações, sem
  nenhuma condição de rolamento defeituoso. Tratado como trabalho futuro
  (ver seção final), não como lacuna de processamento.
- **"Amplitudes e fases das componentes espectrais"**: neste trabalho, "fases"
  refere-se às três fases elétricas do sistema trifásico (A/B/C, isto é,
  tensões `Va/Vb/Vc` e correntes `Ia/Ib/Ic`), já incorporadas desde a primeira
  versão do pipeline — não a ângulo de fase de cada componente harmônica, que
  não está disponível nas planilhas de origem.
- **Modelo de desagregação de cargas**: o resumo descreve duas redes neurais
  recorrentes, uma para diagnóstico e outra para desagregação. O diagnóstico
  de falhas usa de fato uma RNN (Seção 5). A desagregação, por sua vez, foi
  implementada e validada com um modelo de regressão multi-saída baseado em
  Random Forest (`disaggregation.model`), não uma rede recorrente — a base de
  desagregação tem apenas 4 condições combinadas, insuficiente para justificar
  uma arquitetura recorrente sobre uma alternativa mais simples e igualmente
  válida. Os resultados e a discussão tratam a desagregação como um modelo
  supervisionado dedicado (consistente com o objetivo do resumo), sem afirmar
  que é uma rede recorrente. Ver Trabalhos Futuros para a variante recorrente
  como extensão natural caso mais dados combinados sejam coletados.

## 1. Objetivo e perguntas de pesquisa

**Objetivo central**: verificar se, a partir unicamente da assinatura
espectral (componentes harmônicas e interharmônicas de tensão e corrente, nas
três fases), é possível (a) diagnosticar o modo de falha de um sistema
trifásico — normal, falha elétrica isolada, falha mecânica isolada ou falha
combinada — de forma precoce o suficiente para suportar manutenção preditiva,
e (b) desagregar, quando duas cargas operam simultaneamente no mesmo ponto de
acoplamento, a contribuição espectral de cada uma. As duas frentes usam o
mesmo tipo de dado de entrada (assinatura espectral agregada, sem forma de
onda bruta) e são tratadas como objetivos co-primários do trabalho, não como
etapas independentes.

- **RQ-Diagnóstico**: um modelo treinado sobre a assinatura espectral
  consegue classificar corretamente o modo de operação/falha (17 condições —
  saudável, elétrica isolada, mecânica isolada, combinada) com desempenho
  suficiente para suportar detecção precoce de falha (manutenção preditiva)?
- **RQ-NILM**: quando cargas elétricas distintas (máquina + retificador/diodo)
  operam simultaneamente no mesmo ponto de acoplamento comum, é possível
  desagregá-las usando **apenas** a assinatura espectral (harmônicas +
  interharmônicas), sem recorrer à forma de onda bruta?
- **Sub-análise de apoio — sensibilidade mecânica isolada**: como evidência de
  interpretabilidade para o diagnóstico geral, existe relação estatisticamente
  significativa e com tamanho de efeito não desprezível entre componentes
  espectrais específicos e falhas **mecânicas**, quando isoladas do efeito de
  carga não-linear (falha elétrica)? Esta pergunta não é um objetivo da tese
  em si — é um controle metodológico que isola a variável mecânica do efeito
  de confusão elétrico para identificar quais canais espectrais carregam sinal
  mecânico genuíno, fortalecendo a interpretação do diagnóstico geral (RQ-
  Diagnóstico) sobre as condições mecânicas.

**Hipóteses:**
- H-Diagnóstico: um classificador supervisionado (RNN como modelo principal,
  modelos clássicos como baseline) atinge acurácia e F1-macro
  substancialmente acima do acaso nas 17 condições, incluindo discriminação
  entre falha elétrica, mecânica e combinada — não apenas entre saudável e
  falho.
- H-NILM: um modelo supervisionado treinado sobre a assinatura combinada
  consegue recuperar a assinatura da fonte elétrica individual (o
  retificador/diodo) com desempenho consistentemente acima de um preditor
  trivial (predição pela média), avaliado em condições combinadas nunca
  vistas no treino.
- H-Sensibilidade: pelo menos um subconjunto de ordens harmônicas/
  interharmônicas de corrente apresenta separação estatisticamente
  significativa (ANOVA + FDR) e com tamanho de efeito não desprezível
  (η² ≥ 0.06) entre as condições mecânicas puras.

## 2. Desenho experimental — taxonomia de rótulos

O banco de dados (`BANCO_DADOS_CONSOLIDADO.xlsx`, 19.501 aquisições × 2
planilhas, 17 condições experimentais) foi particionado em três grupos
mutuamente exclusivos, codificados em `labels/taxonomy.py` como fonte única de
verdade (reusada tanto no diagnóstico geral quanto na desagregação):

| Categoria | N rótulos | Rótulos | Papel na pesquisa |
|---|---|---|---|
| **Mecânica pura** | 6 | `MÁQUINA BALANCEADA E ALINHADO` (baseline saudável), `MÁQUINA BALANCEADA (S/ PARAF)`, `MÁQUINA DESALINHADA`, `MÁQUINA DESBALANCEADA (RET 6/14 PARAF)`, `MÁQUINA DESBALANCEADA, DESALINHADA (RET 15 PARAF)` | Classe do diagnóstico geral + universo da sub-análise de sensibilidade mecânica |
| **Elétrica pura** | 7 | `DIODO 1`–`DIODO 6`, `RETIFICADOR COMPLETO` | Classe do diagnóstico geral + fonte elétrica isolada para NILM |
| **Combinada** | 4 | `MÁQUINA BALANCEADA E ALINHADA + (-DIODO 2/3/6)`, `... + RETIFICADOR COMPLETO` | Classe do diagnóstico geral + pares treino/teste para NILM |

Os experimentos têm duas origens físicas independentes, ambas monitoradas
pelo mesmo conjunto de analisadores de energia: (1) um retificador trifásico
alimentando carga em corrente contínua, do qual diodos foram progressivamente
removidos para induzir falha elétrica (`DIODO 1`–`6`, `RETIFICADOR COMPLETO`);
(2) uma bancada moto-máquina ajustável, na qual desbalanceamento é induzido
retirando parafusos de um disco calibrado (rótulos `RET N PARAF` — ver Nota de
aderência acima) e desalinhamento é ajustado mecanicamente. As condições
combinadas sobrepõem as duas origens (máquina em falha mecânica ou saudável +
retificador em falha) no mesmo ponto de acoplamento comum.

## 3. Diagnóstico de modos de falha para manutenção preditiva

Universo de análise: as 17 condições experimentais, todas as famílias
espectrais disponíveis — harmônicas (`H`), interharmônicas (`IH`), THD/THDo/
THDe/THD3, K-Factor, TID-F — tensão e corrente, três fases, `avg` e `max`.

1. **Modelo principal**: rede neural recorrente treinada sobre janelas de
   aquisições consecutivas (`deep_learning.architectures`,
   `deep_learning.training`) — comparadas seis arquiteturas (SimpleRNN, LSTM,
   GRU, BiLSTM, CNN+LSTM, Transformer temporal) para justificar a escolha
   final, com ajuste de hiperparâmetros via random search
   (`deep_learning.tuning`).
2. **Baseline clássico**: Random Forest, XGBoost, SVM, MLP, kNN, Regressão
   Logística (`models.classical.compare_models`), sem contexto temporal —
   estabelece o piso de desempenho que a RNN precisa superar para justificar
   a complexidade adicional.
3. **Validação cruzada**: 5-fold estratificado, acurácia e F1-macro por
   classe (17 classes) e agregado por categoria (saudável/elétrica/mecânica/
   combinada).
4. **Interpretabilidade do diagnóstico**: SHAP e permutation importance
   (`sensitivity.importance`) sobre o modelo clássico de melhor desempenho,
   para identificar quais canais espectrais mais contribuem para cada
   classe.
5. **Sub-análise de apoio — sensibilidade mecânica isolada**: restrita às 6
   condições mecânicas puras (Seção 2), para isolar a variável mecânica do
   efeito de confusão elétrico (qualquer harmônico de corrente é, por
   construção, sensível à presença de uma carga não-linear como o
   retificador — testar sensibilidade mecânica num conjunto que já contém
   falha elétrica confundiria as duas causas):
   - ANOVA de um fator por canal com correção FDR
     (`stats.group_tests.sweep_channels`, `test="anova"`,
     Benjamini-Hochberg via `statsmodels.stats.multitest.multipletests`).
   - Tamanho de efeito multi-grupo, η² por canal
     (`sensitivity.ranking.rank_channels_by_eta_squared`) — necessário porque,
     com ~1.400 amostras por condição, quase qualquer diferença real atinge
     significância estatística; η² é o que efetivamente ordena "o quão
     sensível" cada canal é, não apenas "se" há diferença.
   - Tamanho de efeito par-a-par, Cliff's delta não-paramétrico
     (`sensitivity.ranking.rank_channels_by_effect_size`,
     `method="cliffs_delta"`) — falha mecânica específica vs. baseline
     saudável.
   - Classificador 6 classes só nas condições mecânicas
     (`models.classical.compare_models`) com SHAP — se o ranking estatístico
     e o ranking baseado em modelo concordarem, a conclusão sobre quais
     canais carregam sinal mecânico genuíno fica **triangulada por dois
     métodos independentes**, reforçando a interpretação das mesmas classes
     dentro do diagnóstico geral (item 1–2 acima).
   - Mapa de assinatura (`visualization.heatmaps.plot_signature_heatmap`)
     restrito às 6 condições mecânicas, para leitura visual direta.

Ver `notebooks/09_etapa1_mecanica.ipynb` para a execução completa da
sub-análise de sensibilidade contra os dados reais.

## 4. Desagregação de cargas (NILM)

Formulação do problema: dado o vetor de features espectrais (harmônicas +
interharmônicas, tensão e corrente, três fases) de uma amostra do sinal
**combinado** (máquina saudável + retificador/diodo em falha), estimar os
vetores de assinatura das duas fontes individuais.

1. **Dataset supervisionado**: `disaggregation.dataset.build_disaggregation_dataset`
   pareia cada amostra combinada com a assinatura média das duas fontes puras
   correspondentes (`disaggregation.pairs.DISAGGREGATION_PAIRS`, derivado da
   taxonomia da Seção 2) — a precondição que torna isso desagregação
   supervisionada, não separação cega de fontes.
2. **Modelo**: regressão multi-saída (`RandomForestRegressor` via
   `sklearn.multioutput.MultiOutputRegressor`) — um modelo para o alvo
   "máquina", outro para o alvo "fonte elétrica"
   (`disaggregation.model.fit_disaggregation_models`). Ver Nota de aderência
   no início do documento sobre a escolha de Random Forest em vez de uma rede
   recorrente para esta frente.
3. **Validação**: *leave-one-condition-out* — treina nas 3 condições
   combinadas restantes, testa na condição retirada, nunca vista no treino
   (`disaggregation.model.evaluate_disaggregation`,
   `sklearn.model_selection.LeaveOneGroupOut`). Com apenas 4 condições
   combinadas no dataset atual, os resultados devem ser reportados como
   **prova de conceito**, não validação estatisticamente robusta — ver
   Seção 9.
4. **Métricas**: MAE agregado por fold (`evaluate_disaggregation`) e **R² por
   canal individual**, calculado sobre as previsões *out-of-fold agrupadas das
   4 dobras* (`disaggregation.model.per_channel_r2`) — não sobre cada dobra
   isoladamente.
5. **Ressalva estrutural (relevante para os dois alvos, não só "máquina")**:
   cada dobra do *leave-one-out* corresponde a exatamente uma condição
   combinada, então o alvo — seja "máquina" ou "fonte elétrica" — é um vetor
   constante dentro daquela dobra (todas as 4 condições combinadas
   compartilham a mesma máquina-base, e cada dobra tem uma única fonte
   elétrica). Nessa situação, o R² do scikit-learn cai no caso especial de
   variância nula e retorna 0,0 ou 1,0 por definição, não uma medida real de
   ajuste — por isso `evaluate_disaggregation` reporta consistentemente
   R²=1,0 para "máquina" e R²=0,0 para "fonte elétrica" nas 4 dobras: **os
   dois números são artefatos do desenho da validação, não evidência de
   separação (nem de ausência dela)**. O MAE de `evaluate_disaggregation`
   continua válido (não depende de variância do alvo). A métrica que
   efetivamente testa generalização é o R² por canal de `per_channel_r2`,
   porque agrupar as previsões das 4 dobras antes de calcular o R² restaura
   variância real ao alvo (cada condição combinada contribui uma fonte
   elétrica diferente).

Ver `notebooks/10_etapa2_nilm.ipynb` para a execução completa.

## 5. Modelos de Inteligência Artificial empregados

| Modelo | Papel | Módulo |
|---|---|---|
| SimpleRNN, LSTM, GRU, BiLSTM, CNN+LSTM, Transformer temporal | Modelo principal de diagnóstico de falhas (manutenção preditiva) com contexto temporal | `deep_learning.architectures`, `deep_learning.training` |
| Random Forest, XGBoost, SVM, MLP, kNN, Regressão Logística | Baseline de classificação (diagnóstico) e modelo de regressão da desagregação de cargas | `models.classical`, `disaggregation.model` |
| Random search de hiperparâmetros | Ajuste automático (unidades, dropout) para as arquiteturas recorrentes | `deep_learning.tuning` |

## 6. Métricas de desempenho

- **Diagnóstico de falhas**: acurácia, F1-macro (adequado para classes
  desbalanceadas), probabilidade por classe
  (`prediction.fault_classifier.predict_with_probability`).
- **NILM**: R² e MAE, agregados e por canal.
- **Sub-análise de sensibilidade estatística**: p-valor bruto, p-valor
  ajustado por FDR (Benjamini-Hochberg), tamanho de efeito (Cohen's d,
  Cliff's delta, η²) e intervalo de confiança 95% (`stats.confidence`).
- **Previsão de assinatura futura** (exploratório): MSE/MAE normalizados +
  correlação real×previsto (validado em
  `notebooks/08_desagregacao_e_predicao.ipynb`: correlação de 0,996 num canal
  de teste).

## 7. Interpretabilidade

Três fontes de evidência independentes e comparáveis entre si, deliberadamente
triangulados na sub-análise de sensibilidade mecânica (Seção 3, item 5) e
reaproveitadas na leitura do diagnóstico geral:
1. Tamanho de efeito estatístico (η² multi-grupo, Cliff's delta par-a-par).
2. Importância baseada em modelo (SHAP —
   `sensitivity.importance.shap_importance_ranking`, permutation importance —
   `sensitivity.importance.permutation_importance_ranking`).
3. Leitura visual direta (mapas de assinatura, `visualization.heatmaps`).

Quando as três fontes apontam para o mesmo subconjunto de canais, a afirmação
"esses componentes espectrais são os sensíveis à falha X" tem sustentação
metodológica multi-método, não dependente de uma única técnica.

## 8. Contribuição inédita (framing para o artigo)

- Sistema **unificado** de manutenção preditiva e desagregação de cargas a
  partir do mesmo tipo de dado de entrada (assinatura espectral agregada, sem
  forma de onda bruta de alta resolução) — as duas frentes não são estudos
  independentes, mas duas saídas de um mesmo pipeline de aquisição.
- Diagnóstico que cobre falha elétrica, mecânica e combinada no mesmo
  classificador, com uma sub-análise de controle que isola explicitamente o
  efeito mecânico do efeito de carga não-linear antes de generalizar —
  evitando a confusão comum na literatura de MCSA (*Motor Current Signature
  Analysis*) entre assinatura mecânica e assinatura de carga não-linear.
- Comparação lado a lado de **harmônicas e interharmônicas** (não apenas
  harmônicas, mais comum na literatura) como canais diagnósticos.
- Desagregação de cargas (NILM) usando **exclusivamente assinatura espectral
  agregada** (percentuais harmônicos/interharmônicos), sem depender de forma
  de onda bruta de alta resolução temporal — reduz requisito de hardware/
  aquisição frente à maioria da literatura de NILM.
- Metodologia de validação com **tripla triangulação de interpretabilidade**
  (estatística + modelo + visual) aplicada de forma consistente.

## 9. Limitações (a declarar explicitamente no artigo/tese)

- **"Detecção de falhas iminentes" via classificação, não via RUL contínuo**:
  o dataset atual suporta diagnóstico do modo de falha já instalado
  (classificação), o que já atende ao objetivo de manutenção preditiva por
  detecção precoce descrito no resumo. Não suporta, porém, uma curva contínua
  de tempo-até-falha (RUL) — cada rótulo é uma sessão estacionária de ~4–6h
  com a condição já configurada desde o início, sem trajetória contínua de
  degradação com evento de falha registrado. RUL fica como extensão futura
  (ver abaixo), não como parte do escopo atual.
- **Desagregação de cargas não usa rede recorrente**: ver Nota de aderência
  no início do documento — Random Forest multi-saída foi escolhido em vez de
  uma RNN dado o tamanho reduzido do dataset combinado (4 condições).
- **NILM com n pequeno**: apenas 4 condições combinadas, todas compartilhando
  a mesma condição de máquina-base — os resultados de desagregação são prova
  de conceito; para uma validação estatisticamente robusta seria necessário
  coletar mais combinações máquina×falha elétrica (idealmente também variando
  a condição mecânica de base, não só a elétrica).
- **Poder estatístico muito alto na sub-análise de sensibilidade**: com
  ~1.400+ amostras por condição, testes de hipótese isolados tendem a saturar
  em "significativo" — por isso essa sub-análise depende de tamanho de efeito
  e triangulação com modelo, não de p-valor isolado (Seção 3, item 5).

## Trabalhos futuros

- Coleta de dados com **defeitos em rolamentos** na bancada moto-máquina —
  citados no resumo como parte do desenho experimental, ainda não presentes
  no banco de dados atual; extensão direta da taxonomia mecânica (Seção 2).
- Ensaio de degradação contínua (para RUL).
- Mais combinações elétrica×mecânica simultâneas (para NILM robusto e para
  testar se a assinatura mecânica permanece separável mesmo sob falha
  elétrica concorrente).
- Rede neural recorrente dedicada à desagregação de cargas, como evolução do
  modelo Random Forest atual, condicionada a uma base de condições combinadas
  maior (ver limitação acima).
- Ajuste de hiperparâmetros mais extenso das arquiteturas recorrentes
  (`deep_learning.tuning.random_search` já implementado, usado aqui só como
  prova de conceito com poucos épocos/tentativas).
