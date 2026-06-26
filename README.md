# Predição de Internações Prolongadas no SUS com Machine Learning

## Visão geral

Este projeto tem como objetivo construir um modelo de Machine Learning capaz de identificar internações hospitalares com maior risco de permanência prolongada no SUS, utilizando dados públicos do Sistema de Informações Hospitalares do SUS — SIH/SUS.

A proposta é demonstrar como dados públicos de saúde podem ser utilizados para apoiar análises preditivas, auxiliando na identificação antecipada de internações com maior probabilidade de longa permanência. Esse tipo de informação pode contribuir para a gestão hospitalar, planejamento de leitos e priorização de acompanhamento de pacientes.

O projeto foi desenvolvido em Python, utilizando etapas de coleta, tratamento, análise exploratória, modelagem, ajuste de hiperparâmetros e validação temporal.

---

## Problema

Internações prolongadas podem impactar diretamente a ocupação de leitos, os custos hospitalares e a capacidade de atendimento do sistema público de saúde. Em hospitais com alta demanda, identificar previamente quais internações possuem maior risco de longa permanência pode ser útil para apoiar decisões administrativas e estratégias de gestão.

A pergunta central do projeto é:

> É possível utilizar dados administrativos do SIH/SUS para prever quais internações têm maior risco de se tornarem prolongadas?

---

## Objetivo

Construir um modelo de classificação binária para prever internações prolongadas no SUS.

A variável-alvo foi construída a partir da coluna `DIAS_PERM`, classificando as internações em:

* `0`: internação não prolongada;
* `1`: internação prolongada.

O foco principal da modelagem foi a classe `1`, pois ela representa os casos de maior interesse para o problema.

---

Fonte dos dados

Os dados utilizados são públicos e foram obtidos a partir do SIH/SUS, disponibilizado pelo DATASUS.

Foram utilizados registros de internações hospitalares do estado de Mato Grosso entre 2020 e 2024.

Devido ao tamanho dos arquivos, os dados brutos e/ou tratados podem não estar versionados diretamente neste repositório.

Link para acesso aos dados utilizados: [https://drive.google.com/file/d/1asi8ClWYtRe-7Ji-FAMgALZ-de3P-cN-/view?usp=sharing]

---

## Estrutura do repositório

```text
SUS/
│
├── dados/
│   ├── raw/                     # Dados brutos
│   ├── interim/                 # Dados intermediários
│   └── processed/               # Dados tratados para modelagem
│       └── splits/              # Bases separadas por janelas temporais
│
├── 01 - Criação da base.ipynb
├── 02 - Pré-processamento.ipynb
├── 03 - Modelagem - Comparação dos modelos.ipynb
├── 04 - Modelagem - Duelo dos 3 melhores.ipynb
├── 05 - Modelagem - Ajustes de hiperparametros.ipynb
├── 06 - Validação Temporal e Modelo Final.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Metodologia

O projeto foi organizado em etapas progressivas:

1. **Criação da base**

   * Coleta dos arquivos do SIH/SUS;
   * Conversão dos arquivos originais;
   * Consolidação dos dados em uma base única.

2. **Pré-processamento**

   * Seleção de variáveis;
   * Tratamento de tipos;
   * Criação da variável-alvo;
   * Criação de variáveis derivadas;
   * Codificação de variáveis categóricas;
   * Separação temporal da base.

3. **Modelagem inicial**

   * Comparação entre diferentes algoritmos;
   * Avaliação dos modelos com e sem SMOTE;
   * Seleção dos três melhores modelos.

4. **Ajuste de hiperparâmetros**

   * Ajustes básicos dos modelos selecionados;

5. **Refinamento do modelo vencedor**

   * Refinamento do modelo vencedor com `RandomizedSearchCV`;
   * Escolha da melhor versão da Regressão Logística.

6. **Validação temporal**

   * Avaliação com janela expansiva;
   * Treino em anos anteriores e teste no ano seguinte;
   * Verificação da estabilidade do modelo ao longo do tempo.

---

## Variáveis utilizadas

As principais variáveis utilizadas no modelo foram relacionadas a características da internação, idade, período, hospital, município, especialidade, diagnóstico e procedimento.

Entre as variáveis utilizadas ou derivadas estão:

* `idade_anos`;
* `mes_internacao`;
* `MUNIC_MOV`;
* `CNES`;
* `ESPEC`;
* `cid3`;
* `grupo_procedimento`;
* `internacao_prolongada`.

A variável `DIAS_PERM` foi utilizada apenas para construção da variável-alvo e não foi utilizada como preditora, para evitar vazamento de informação.

Variáveis sensíveis ou de uso mais delicado, como sexo, raça/cor e etnia, não foram priorizadas no modelo final, considerando possíveis riscos de viés e limitações éticas na aplicação de modelos preditivos em saúde.

---

## Modelos avaliados

Foram avaliados seis modelos iniciais:

1. Regressão Logística sem SMOTE;
2. Regressão Logística com SMOTE;
3. Random Forest sem SMOTE;
4. Random Forest com SMOTE;
5. XGBoost sem SMOTE;
6. XGBoost com SMOTE.

Os modelos foram avaliados principalmente pelas métricas da classe `1`, que representa as internações prolongadas:

* `recall`;
* `precision`;
* `f1-score`;
* matriz de confusão.

A acurácia foi analisada como métrica complementar, mas não foi utilizada isoladamente para escolha do modelo, pois a base possui desbalanceamento entre as classes.

---

## Seleção do modelo

Após a comparação inicial, os três modelos com melhor desempenho geral foram:

* Regressão Logística com SMOTE;
* Random Forest com SMOTE;
* XGBoost com SMOTE.

Após ajustes básicos de hiperparâmetros, a Regressão Logística com SMOTE apresentou o melhor equilíbrio geral entre recall, precision e f1-score.

Em seguida, a Regressão Logística foi refinada com `RandomizedSearchCV`, testando diferentes valores de regularização e solvers.

O modelo final escolhido foi:

```python
LogisticRegression(
    solver="saga",
    penalty="l2",
    C=5.0,
    max_iter=3000,
    tol=0.0001,
    random_state=42
)
```

O SMOTE foi aplicado apenas nas bases de treino, após a padronização das variáveis numéricas.

---

## Resultado da validação temporal

Para avaliar a estabilidade do modelo ao longo do tempo, foi utilizada validação temporal com janela expansiva.

As janelas avaliadas foram:

| Janela temporal               | Accuracy | Recall | Precision | F1-score |
| ----------------------------- | -------: | -----: | --------: | -------: |
| Treino 2020 → Teste 2021      |   76,83% | 70,83% |    50,15% |   58,72% |
| Treino 2020–2021 → Teste 2022 |   77,87% | 70,15% |    47,91% |   56,93% |
| Treino 2020–2022 → Teste 2023 |   77,77% | 68,63% |    47,24% |   55,96% |
| Treino 2020–2023 → Teste 2024 |   77,27% | 70,71% |    46,02% |   55,75% |

---

## Interpretação dos resultados

A validação temporal mostrou que o modelo manteve desempenho relativamente estável ao longo dos anos avaliados.

A acurácia permaneceu próxima de 77% em todas as janelas, enquanto o recall da classe `1` ficou próximo de 70%. Isso indica que o modelo conseguiu preservar sua capacidade de identificar internações prolongadas mesmo quando aplicado a anos futuros.

No entanto, a precision apresentou queda gradual ao longo das janelas, passando de aproximadamente 50,15% em 2021 para 46,02% em 2024. Isso mostra que o modelo ainda gera uma quantidade relevante de falsos positivos.

Em termos práticos, o modelo é capaz de identificar boa parte das internações prolongadas, mas também classifica algumas internações não prolongadas como prolongadas.

---

## Conclusão

O modelo final apresentou desempenho estável e conseguiu identificar aproximadamente 70% das internações prolongadas nas diferentes janelas temporais.

A Regressão Logística com SMOTE foi escolhida como modelo final por apresentar bom equilíbrio entre desempenho, interpretabilidade e custo computacional. Além disso, a validação temporal indicou que o modelo mantém capacidade de generalização ao longo dos anos.

Apesar disso, o modelo não deve ser interpretado como uma ferramenta de decisão automática. Seu uso mais adequado seria como uma ferramenta de apoio à triagem e à gestão hospitalar, sinalizando internações com maior risco de longa permanência para análise posterior.

---

## Limitações

Algumas limitações do projeto incluem:

* uso de dados administrativos, que podem conter inconsistências ou mudanças de padrão de preenchimento;
* ausência de variáveis clínicas mais detalhadas;
* uso de dados de apenas um estado;
* presença relevante de falsos positivos;
* possível variação dos padrões hospitalares ao longo dos anos;
* necessidade de validação com especialistas da área de saúde antes de qualquer aplicação prática.

---

## Possíveis melhorias futuras:

* testar diferentes thresholds de decisão para melhorar o equilíbrio entre recall e precision;
* avaliar outros estados ou regiões do Brasil;
* incluir novas variáveis clínicas e administrativas;
* testar modelos calibrados probabilisticamente;
* comparar o desempenho por hospital, município ou especialidade;
* construir um dashboard com indicadores de risco e resultados do modelo;
* salvar e disponibilizar o modelo final em formato reutilizável.

---

## Tecnologias utilizadas

* Python;
* Pandas;
* NumPy;
* Scikit-learn;
* Imbalanced-learn;
* XGBoost;
* Matplotlib;
* Seaborn;
* Jupyter Notebook;
* Git/GitHub.

---

## Autor

Projeto desenvolvido por **Djonathan Cezar de Souza Metelo**.

LinkedIn: [Djonathan Metelo](https://www.linkedin.com/in/djonathan-metelo/)
GitHub: [DjonathanMetelo](https://github.com/DjonathanMetelo)


---

## Observação

Este projeto possui finalidade educacional e de portfólio. Os resultados apresentados não devem ser utilizados isoladamente para tomada de decisão clínica, hospitalar ou administrativa sem validação técnica, estatística e especializada.
