###########################################################################################
# Weighted k-Nearest Neighbors - wkNN (Vizinho mais próximo ponderado por funções kernel)
###########################################################################################
# Funciona razoavelmente bem para problemas
# de baixa dimensão...

## 1: Instala pacotes necessários -----------------------------------------
# install.packages(c("caret", "data.table", "tidyverse", "kknn", "easypackages"))

## 2: Carrega pacotes necessários -----------------------------------------
easypackages::libraries("caret","data.table",
                        "tidyverse","kknn")

## 3: Carrega conjunto de dados -------------------------------------------
data <- fread("./docs/R-scripts/Tectona.csv", stringsAsFactors=T)

## 4: Análise Exploratória dos Dados - EAD
### 4.1: Estatística descritiva
### 4.2: Análise de correlação
### 4.3: Relação entre variáveis

## 4.3: Relação entre variáveis
ggplot(melt(data, id.vars=c("V")),
       aes(x=value, y=V)) + geom_point() +
  facet_grid( ~ variable)

## 5: Engenharia de recursos (feature engineering) ----------------------
# data[,D2:=D^2
#      ][,lnD:=log(D)
#        ][,invD:=1/D
#          ][,D2H:=D^2*H
#            ][,lnH:=log(H)
#              ][,DH2:=D*H^2
#                ][,lnD2H:=log(D^2*H)
#                  ][,H2:=H^2
#                    ][,DH:=D*H]

# Obs.: Alguns algoritmos de AM dispensam transformações
# (ex.: dummy, one-hot-encoding). As RNAs, por exemplo,
# requerem normalização de variáveis e transformações para
# variáveis qualitativas. Para k-NN é necessário transformações
# de qualitativas (se existentes), pois o algoritmo usa
# métricas de distâncias.

## 6: Divisão aleatória estratificada ------------------------------------
set.seed(100)
trainIndex <- createDataPartition(y=data$V, p=.70, list=FALSE)
trainingSet <- data[trainIndex,]
testSet <- data[-trainIndex,]

## 7: Configuração do treinamento ----------------------------------------
source("./docs/R-scripts/Summary.R")

fitControl <- trainControl(method = "LOOCV",
                           summaryFunction = Summary,
                           verboseIter = T,
                           selectionFunction = "best")

### 8.1: Hiperparâmetros candidatos --------------------------------------
# Estratégia grid search...
# 3 hiperparâmetros de ajuste...
# distance = Euclidiana
?kknn
tuneGrid <- expand.grid(kmax = seq(1,10,1),
                        kernel = c("gaussian", "rectangular"),
                        distance = 2)

### 7.2: Ajuste de hiperparâmetros (LOOCV)
# Escalonar e centralizar (ou normalizar) as variáveis
# numericas no k-NN (e suas variações), é importante
# para diminuir os efeitos de variáveis com maiores
# escalas de medidas sobre a determinação das métricas
# de distâncias.

# normalização z-score ("center","scale")
# (xi - xbar)/sd(x)

set.seed(1000)
m_knn <- train(V ~.,
               data = trainingSet,
               method = "kknn",
               tuneGrid = tuneGrid,
               preProcess = c("center","scale", "BoxCox"),
               trControl = fitControl)

# Melhor configuração
m_knn$bestTune

# Indica dentre todas as configurações avaliadas aquela com menor
# erro na validação LOOCV. São os "Hiperparâmetros tuning"...

# Gráfico de desempenho usando LOOCV
ggplot(m_knn) +
  geom_vline(xintercept=m_knn$bestTune[1]$kmax,
             colour = "red", linetype="dotted") +
  scale_x_continuous(limits = c(1,10),
                     breaks=seq(1,10,1),
                     "kmax") +
  theme_bw()

# Desempenho médio na validação LOOCV
results.m_knn <- m_knn$results
setorder(results.m_knn, RMSE)

# Salva e ler os modelos
saveRDS(m_knn,'./docs/models/m_knn.rds')
m_knn <- readRDS('./docs/models/m_knn.rds')

# Desempenho no conjunto de teste (30%)
(pred <- predict(m_knn, testSet))
df <- data.frame(obs = testSet$V, pred = pred)
round(Summary(data=df), 4)

# Como disponibilizar?
###########################################################################################
# Ok! Mas, como disponibilizar o modelo para um usuário
# final?

# Opção 1: Enviar o arquivo "m_knn.rds" para quem
# deseja usar o modelo.

# Problema: E os não usuários de R?

# Opção 2: Disponibilizar através de uma aplicação.
