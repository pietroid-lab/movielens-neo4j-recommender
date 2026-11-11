# MovieLens Graph Recommender — Neo4j + Pearson + KNN

Projeto acadêmico de recomendação colaborativa de filmes, usando somente `movies.csv` e `ratings.csv` do MovieLens. O cálculo de Pearson, a previsão KNN e a avaliação são implementados em Python; o Neo4j armazena o grafo e executa as consultas de recomendação.

## 1. Modelagem do grafo

```text
(:User {userId})-[:RATED {rating, timestamp}]->(:Movie {movieId, title, genres})
(:Movie)-[:SIMILAR {correlation, commonUsers}]->(:Movie)
```

- **User**: `userId`, originado em `ratings.csv`.
- **Movie**: `movieId`, `title` e `genres`, originados em `movies.csv`.
- **RATED**: `rating` e `timestamp`, originados em `ratings.csv`.
- **SIMILAR**: `correlation` de Pearson e `commonUsers`, calculados apenas a partir das avaliações de treino. Cada par de filmes é armazenado uma vez, orientado pelo menor `movieId` para o maior; consultas podem percorrer o relacionamento em qualquer direção.

## 2. Requisitos

- Python 3.10 ou superior.
- Docker Desktop (ou Docker Engine + Docker Compose).
- O driver oficial `neo4j` (única dependência Python externa deste projeto).
- Conexão à internet na primeira execução para baixar o MovieLens latest-small.

O MovieLens latest-small contém cerca de 100 mil ratings e 9,7 mil filmes. Fonte oficial e descrição: <https://grouplens.org/datasets/movielens/>. O downloader extrai somente `movies.csv` e `ratings.csv`.

## 3. Instalação no Windows (PowerShell) ou Linux/macOS

Na pasta do projeto:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
cp .env.example .env
pip install -r requirements.txt
```

Se usar o PowerShell e a política bloquear a ativação do ambiente virtual, execute os comandos com `python -m pip` usando o Python da `.venv`, ou ajuste a política conforme as regras da sua máquina.

## 4. Iniciar o Neo4j

Com Docker ativo:

```bash
docker compose up -d
```

- Browser Neo4j: <http://localhost:7474>
- Bolt: `bolt://localhost:7687`
- Usuário padrão: `neo4j`
- Senha padrão local: `moviegraph123` (configurada em `.env.example`)

Se alterar `NEO4J_PASSWORD` no `.env`, o Docker Compose usa esse valor quando o volume for inicializado pela primeira vez. Se o banco já tiver sido inicializado com outra senha, alterar o arquivo não troca automaticamente a senha persistida.

## 5. Pipeline completo

Execute na ordem abaixo. O primeiro comando integrado pode ser usado para a pipeline inteira:

```bash
python main.py all --reset-graph
```

Esse comando baixa o dataset se necessário, cria o split, calcula Pearson e importa o grafo. **Atenção:** `--reset-graph` apaga todos os nós e relacionamentos do banco Neo4j configurado. Use um banco dedicado a este projeto.

Para executar etapa por etapa:

```bash
python main.py download
python main.py prepare --test-size 0.2 --seed 42
python main.py similarities --min-common 3
python main.py import --reset-graph
```

Os artefatos gerados ficam em `data/processed/`:

- `ratings_train.csv`: apenas avaliações usadas para treino.
- `ratings_test.csv`: avaliações reservadas para teste.
- `similarities.csv`: correlações calculadas somente com o treino.
- `evaluation_results.csv`: métricas por usuário, geradas durante a avaliação.

A separação é aleatória, reproduzível e global, com 80% das linhas no treino e 20% no teste. O banco recebe apenas `ratings_train.csv`. Portanto, as notas de teste não participam do cálculo das similaridades nem são inseridas no grafo.

## 6. Consultas solicitadas

### Quantas comédias o usuário avaliou com nota 4 ou superior?

```bash
python main.py comedies --user-id 1 --min-rating 4
```

A consulta verifica `Comedy` dentro dos gêneros separados por `|` e conta filmes distintos.

### Top recomendações para o usuário X

```bash
python main.py recommend --user-id 1 --k 20 --limit 10
```

`k=20` usa os 20 vizinhos mais correlacionados disponíveis por candidato; `limit=10` devolve o top 10. O `user-id=1` é apenas exemplo. Troque pelo ID que deseja consultar.

## 7. Como a similaridade é calculada

Para cada par de filmes, são usados somente os usuários que avaliaram **ambos** os filmes. A correlação de Pearson é calculada a partir das somas suficientes para o conjunto de co-avaliadores. Pares com variância zero não têm correlação definida e são descartados. Por padrão, exige-se um mínimo de três usuários em comum (`--min-common 3`); esse filtro reduz relações sustentadas por evidência muito escassa. O número exato de co-avaliadores é armazenado em `commonUsers`.

A correlação pode ser negativa. Para previsão, a soma ponderada usa a correlação assinada no numerador e o valor absoluto no denominador:

```text
prediction(u, i) = sum(correlation(i,j) * rating(u,j)) /
                   sum(abs(correlation(i,j)))
```

São usados no máximo `k` vizinhos já avaliados pelo usuário; filmes que o usuário já avaliou não são recomendados. A previsão final é limitada ao intervalo de notas `[0.5, 5.0]` do MovieLens.

## 8. Avaliação (Precision@K e Recall@K)

```bash
python main.py evaluate --k 20 --top-n 10 --threshold 4
```

Uma nota no teste igual ou superior a `4` é tratada como relevante. Para cada usuário com ao menos um item relevante no conjunto de teste:

- `Precision@K = hits / K`
- `Recall@K = hits / total de itens relevantes desse usuário no teste`

O resultado apresentado é a média macro das métricas por usuário. O denominador de `Precision@K` é sempre `K`, mesmo que o sistema encontre menos de K recomendações. Resultados detalhados por usuário são gravados em `data/processed/evaluation_results.csv`.

A avaliação consulta recomendações no grafo que foi construído exclusivamente com o treino. Se um usuário não tiver recomendações, suas métricas são zero. Usuários sem nenhum item relevante no teste são excluídos da média porque o recall seria indefinido. Os hiperparâmetros e o limiar de relevância devem ser mencionados no relatório.

## 9. Testes locais

Os testes usam apenas a biblioteca `unittest` e dados sintéticos; não exigem Neo4j, dataset ou bibliotecas extras:

```bash
python -m unittest discover -s tests -v
```

Eles verificam Pearson positivo/negativo e casos indefinidos, previsão ponderada, limites das notas, precisão/recall e geração do split.

## 10. Resolução de problemas

- **Connection refused / erro Bolt**: execute `docker compose ps` e `docker compose logs neo4j`; confirme a porta `7687` e as credenciais em `.env`.
- **Authentication failed**: confira a senha persistida no volume do Neo4j. A senha do volume existente não muda apenas editando `.env`.
- **Nenhuma recomendação**: confirme que o grafo foi importado, que o usuário tem avaliações no treino e que existem similaridades com correlação diferente de zero.
- **Arquivo não encontrado**: execute primeiro `python main.py download`, `prepare` e `similarities`.
- **Importação interrompida**: reexecute `python main.py import --reset-graph` para reconstruir o banco dedicado a este projeto.

## 11. Atribuição do dataset

Harper, F. Maxwell, and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4), 19:1–19:19. <https://doi.org/10.1145/2827872>.

Confira a licença e as condições de uso do dataset na página oficial do GroupLens antes de redistribuir o conjunto de dados ou publicar resultados.
