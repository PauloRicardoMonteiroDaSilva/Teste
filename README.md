# Projeto de Classificação de Tipos de Violência - Trabalho Final

**Disciplina:** Engenharia de Sistemas Inteligentes (CK0444)
**Universidade:** Universidade Federal do Ceará (UFC)
**Autor:** Paulo Ricardo
**Semestre:** 2025.1

---

## 1. Visão Geral do Projeto

Este projeto representa a implementação completa de um sistema de Machine Learning, desenvolvido como trabalho final da disciplina de Engenharia de Sistemas Inteligentes. O objetivo é construir um serviço preditivo capaz de analisar as circunstâncias de uma notificação de violência e **classificar os tipos de violência mais prováveis** (física, psicológica, sexual), cumprindo todas as etapas de um pipeline MLOps.

O sistema foi estruturado em três módulos principais, conforme especificado nos requisitos do trabalho:

1.  **Módulo de Pipeline de Dados:** Responsável pela ingestão, limpeza e transformação dos dados.
2.  **Módulo de Pipeline de Modelos:** Responsável pelo treino, comparação e serialização de um modelo de classificação.
3.  **Módulo de Serviço:** Uma API RESTful que expõe o modelo treinado para previsões em tempo real.

### Adaptação da Tarefa de Machine Learning

Durante a análise exploratória, foi constatado que a tarefa inicial de "prever reincidência" era inviável, pois o conjunto de dados não continha exemplos suficientes de casos de "não reincidência". Demonstrando a adaptabilidade necessária em projetos reais de ciência de dados, o projeto foi "pivotado" com sucesso para uma tarefa mais complexa e viável: **classificação multi-rótulo (multi-label) dos tipos de violência**.

---

## 2. Estrutura de Pastas do Projeto

O projeto segue uma estrutura padronizada para facilitar a organização, manutenção e escalabilidade.


/projeto-violencia-ml/
│
├── 📄 Dockerfile               <- Instruções para criar o container da API
├── 📄 README.md                <- Este arquivo de documentação
├── 📄 requirements.txt          <- Lista de dependências Python do projeto
│
├── 📁 data/
│   ├── 📁 raw/                  <- Local para os 15 arquivos CSV originais
│   └── 📁 processed/            <- Dados limpos e unificados, gerados pelo pipeline
│
├── 📁 notebooks/
│   └── 🐍 analise_exploratoria.py <- Script para gerar gráficos e análises (usar em Jupyter)
│
├── 📁 src/
│   ├── 🐍 data_pipeline.py     <- Script para combinar e limpar os dados brutos
│   ├── 🐍 model_pipeline.py    <- Script para treinar e salvar o modelo
│   └── 📁 api/
│       ├── 🐍 app.py           <- Código da API com FastAPI
│       ├── 📦 model.pkl        <- Arquivo do modelo treinado (gerado)
│       └── 📦 model_columns.pkl  <- Arquivo das colunas do modelo (gerado)
│
└── 📁 tests/
└── 🐍 test_api.py          <- Testes automatizados para a API


---

## 3. Como Executar o Projeto

Siga os passos abaixo para configurar e executar o projeto completo no seu ambiente local.

### Pré-requisitos

* Python 3.8 ou superior
* `pip` (gerenciador de pacotes do Python)
* Docker (opcional, para a execução via container)

### Passo a Passo

**1. Preparação do Ambiente**

* **Crie a estrutura de pastas** conforme o diagrama acima.
* **Crie e ative um ambiente virtual** (altamente recomendado):
    ```bash
    # No terminal, na pasta raiz do projeto
    python -m venv venv
    
    # Ativar no Windows
    .\venv\Scripts\activate
    ```
* **Instale as dependências**:
    ```bash
    pip install -r requirements.txt
    ```

**2. Organize os Dados**

* Coloque todos os 15 arquivos CSV de dados de violência dentro da pasta `data/raw/`.

**3. Execute os Pipelines em Ordem**

* Navegue até a pasta `src` no seu terminal: `cd src`
* **Execute o Pipeline de Dados:**
    ```bash
    python data_pipeline.py
    ```
* **Execute o Pipeline do Modelo:**
    ```bash
    python model_pipeline.py
    ```

**4. Execute a API (Opção A: Localmente)**

* Navegue até a pasta da API: `cd api`
* **Inicie o servidor da API:**
    ```bash
    python -m uvicorn app:app --reload
    ```
* A API estará disponível em [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

**5. Execute os Testes Automatizados**

* Na pasta raiz do projeto, execute o `pytest`:
    ```bash
    pytest
    ```

---

## 4. Execução com Docker (Avançado)

A containerização garante que a aplicação funcione em qualquer ambiente.

**1. Construa a Imagem Docker**

* Na pasta raiz do projeto (onde está o `Dockerfile`), execute:
    ```bash
    docker build -t violencia-api .
    ```

**2. Execute o Container**

* Após a imagem ser construída, inicie um container:
    ```bash
    docker run -d -p 8000:8000 --name violencia-container violencia-api
    ```
* A API estará disponível no seu navegador em [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## 5. Explicação Detalhada do Código

### 5.1. `src/data_pipeline.py`

Este script transforma os dados brutos e inconsistentes num único dataset limpo e pronto para a modelagem.

* **Estratégia de Memória e Codificação:** Para evitar erros de memória, o script lê os arquivos CSV em "pedaços" (`chunks`). Ele utiliza a codificação `utf-8-sig` para remover automaticamente o caractere especial invisível (BOM, `ï»¿`) do início dos ficheiros, garantindo que os nomes das colunas sejam lidos corretamente.
* **Engenharia de Atributos:** Cria novas features para enriquecer o modelo:
    * `idade_calculada`: Uma idade mais precisa, calculada a partir da data de nascimento e da data de notificação.
    * `dia_semana` e `mes_notificacao`: Features temporais que podem ajudar o modelo a capturar padrões sazonais.

### 5.2. `src/model_pipeline.py`

Este script é responsável por toda a lógica de Machine Learning para a tarefa de **classificação de tipo de violência**.

* **Preparação para Multi-label:** A função `prepare_data_for_violence_classification` prepara os dados para a nova tarefa. As colunas `violencia_fisica`, `violencia_psicologica` e `violencia_sexual` são definidas como o alvo (`y`), e as outras características (idade, local, etc.) são definidas como as features preditoras (`X`).
* **Modelo Multi-Output:** O script utiliza um `RandomForestClassifier` envolvido por um `MultiOutputClassifier`. Esta abordagem permite que um único modelo aprenda a prever múltiplos resultados (os diferentes tipos de violência) em simultâneo.
* **Avaliação Detalhada:** O desempenho do modelo é avaliado de forma granular. Um relatório de classificação é gerado para cada tipo de violência, permitindo uma análise precisa de onde o modelo acerta e erra para cada categoria.

### 5.3. `src/api/app.py`

Este script expõe o modelo treinado através de uma API, tornando-o utilizável.

* **Endpoint `/predict`:** Recebe uma requisição `POST` com as características de uma notificação (idade, local, etc.).
* **Lógica de Previsão:**
    1.  Adiciona automaticamente as features temporais (dia e mês atuais) aos dados recebidos.
    2.  Aplica as mesmas transformações (One-Hot Encoding) que foram feitas nos dados de treino.
    3.  Usa o modelo `model.pkl` carregado para prever a probabilidade de cada tipo de violência.
    4.  Retorna uma resposta JSON clara, indicando se cada tipo de violência é provável ou não.

* **Endpoint `/docs`:** O FastAPI gera automaticamente uma documentação interativa (Swagger UI), que serve como uma interface gráfica para testar e demonstrar o funcionamento da API.
