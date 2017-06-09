# Aprendizaje Automático - Proyecto Final
# Autores:
#  Nuria Rodríguez Barroso
#  Juan Luis Suárez Díaz

    # Función para pausar la ejecución
    pause <- function(){
      readline("Presione [Intro] para continuar...")
    }

    #Fijamos la semilla para obtener siempre los mismos resultados
    set.seed(123456789)

    #Añadimos librerías necesarias.
    library("caret")
    library("e1071")
    library("glmnet")
    library("leaps")
    library("DMwR")
    library("neuralnet")
    library("nnet")
    library("randomForest")
    library("gbm")
    library("adabag")
    library("maboost")


    #LECTURA DE LOS DATOS
    har.train <- read.table("./datos/X_train.txt", sep="", head = F)
    lhar.train <- read.table("./datos/y_train.txt", sep="", head = F)
    har.data.train <- cbind(lhar.train, har.train)
    
    har.test <- read.table("./datos/X_test.txt", sep="", head = F)
    lhar.test <- read.table("./datos/y_test.txt", sep="", head = F)
    har.data.test <- cbind(lhar.test, har.test)


    print(summary(har.train[,1:10]))
    
    pause()

  #Estudiamos si hay desbalanceo
  print("El porcentaje de elementos con etiqueta 1 en el train es: ")
  print(100*sum(lhar.train == 1)/nrow(lhar.train))
  print("El porcentaje de elementos con etiqueta 2 en el train es: ")
  print(100*sum(lhar.train == 2)/nrow(lhar.train))
  print("El porcentaje de elementos con etiqueta 3 en el train es: ")
  print(100*sum(lhar.train == 3)/nrow(lhar.train))
  print("El porcentaje de elementos con etiqueta 4 en el train es: ")
  print(100*sum(lhar.train == 4)/nrow(lhar.train))
  print("El porcentaje de elementos con etiqueta 5 en el train es: ")
  print(100*sum(lhar.train == 5)/nrow(lhar.train))
  print("El porcentaje de elementos con etiqueta 6 en el train es: ")
  print(100*sum(lhar.train == 6)/nrow(lhar.train))

  pause()

  print("El porcentaje de elementos con etiqueta 1 en el test es: ")
  print(100*sum(lhar.test == 1)/nrow(lhar.test))
  print("El porcentaje de elementos con etiqueta 2 en el test es: ")
  print(100*sum(lhar.test == 2)/nrow(lhar.test))
  print("El porcentaje de elementos con etiqueta 3 en el test es: ")
  print(100*sum(lhar.test == 3)/nrow(lhar.test))
  print("El porcentaje de elementos con etiqueta 4 en el test es: ")
  print(100*sum(lhar.test == 4)/nrow(lhar.test))
  print("El porcentaje de elementos con etiqueta 5 en el test es: ")
  print(100*sum(lhar.test == 5)/nrow(lhar.test))
  print("El porcentaje de elementos con etiqueta 6 en el test es: ")
  print(100*sum(lhar.test == 6)/nrow(lhar.test))
  
  pause()
    

   # Asimetría de los datos
   #Ordenamos las columnas por asimetria
    har_asymmetry <- apply(har.train, 2, skewness)
    har_asymmetry <- sort(abs(har_asymmetry), decreasing = T)
    print(head(har_asymmetry))
    

  hist(har.train$V389, col = "blue")

  pause()
  
## ---- PCA --------------------------------------------------
set.seed(123456789)
ObjetoTrans <- preProcess(har.train,method = c("pca"), thresh = 0.95)


  print(ObjetoTrans)
  
  pause()

## ---- LASSO --------------------------------------------------
    set.seed(123456789)

    ml_lasso <- cv.glmnet(as.matrix(har.train), lhar.train[,1], family="multinomial", nfolds = 3)
    lambda.min <- ml_lasso$lambda.min
    lambda.1se <- ml_lasso$lambda.1se
    
    coeffs.min <- coef(ml_lasso, s = ml_lasso$lambda.min)
    coeffs.1se <- coef(ml_lasso, s = ml_lasso$lambda.1se)
    
    #print(coeffs.min)
    #print(coeffs.1se)
    
    #Eliminamos los atributos del train  

    coefficients.min <- matrix(ncol = 6, nrow = 562)
    coefficients.1se <- matrix(ncol = 6, nrow = 562)
    for(i in 1:6){
      coefficients.min[,i] <- (coeffs.min[[i]])[1:nrow(coeffs.min[[i]]),]
      coefficients.1se[,i] <- (coeffs.1se[[i]])[1:nrow(coeffs.1se[[i]]),]
    }

## ------------------------------------------------------
    
    har.train_lasso.min = har.train[,abs(coefficients.min[1:6,])>0]
    har.train_lasso.1se = har.train[,abs(coefficients.1se[1:6,])>0]
    
    print("El número de atributos tras aplicar la reducción con lambda.min es: ")
    print(dim(har.train_lasso.min)[2])
    
    print("El número de atributos tras aplicar la reducción con lambda.1se es: ")
    print(dim(har.train_lasso.1se)[2])

    pause()

## ---- Transformamos los datos con PCA --------------------------------------------------
har.trans.train <- predict(ObjetoTrans, har.train)
har.trans.test <- predict(ObjetoTrans, har.test)

har.train <- har.trans.train  
har.test <- har.trans.test

#### MODELO LINEAL

## ------------------------------------------------------
calculateEtest <- function(ml, test, ltest, s=0){

  #Cálculo de probabilidades
  if(s==0){
    ml.prob_test = predict(ml, test, type="response")
  }else{
    ml.prob_test = predict(ml,as.matrix(test),type = "response", s = s)
  }

  # Etest
  ml.pred_test = rep(0, length(ml.prob_test)) # predicciones por defecto 0
  ml.pred_test[ml.prob_test >=0.5] = 1 # >= 0.5 clase 1
  

  ml.Etest = mean(ml.pred_test != ltest)
  
  return(list(ml.Etest, ml.prob_test,ml.pred_test))
}

calculateEin <- function(ml ,ltrain){
  ml.prob_train = predict(ml, type = "response") #no tenemos que introducirle el train porque recuerda
  # Ein
  ml.pred_train = rep(0, length(ml.prob_train)) #predicciones por defecto 0
  ml.pred_train[ml.prob_train >= 0.5] = 1
    
  ml.Ein = mean(ml.pred_train != ltrain)

  return(list(ml.Ein, ml.prob_train,ml.pred_train))
}

testFamilies <- function(models, test, ltrain, ltest, modelNames = NULL){

  results <- matrix(nrow=length(models), ncol = 2)
  colnames(results) <- c("Ein","Eout")
  if(!is.null(modelNames)){
    rownames(results) <- modelNames
  }
  for(i in seq_along(models)){
    results[i,1] <-calculateEin(models[[i]],ltrain)[[1]]
    results[i,2] <- 
      calculateEtest(models[[i]],test,ltest)[[1]]
  }
  
  return(results)
}

## ---- Predicción 1 vs 1---------------------------------------------------
predictions1vs1 <- function(train, test, ltrain, ltest, modelNames = NULL){
  
  preds_ij <- matrix(nrow = 15, ncol = length(ltest))
  
  k <- 0
  
  for(i in 2:6){
    for(j in 1:(i-1)){
      #cat(i," vs ",j,":\n")
      
      inds_i <- which(ltrain == i)
      inds_j <- which(ltrain == j)

      ltrain_i <- ltrain[inds_i]
      ltrain_j <- ltrain[inds_j]
      
      train_i <- train[inds_i,]
      train_j <- train[inds_j,]
      
      
      train_ij <- rbind(train_i,train_j)
      ltrain_ij <- c(ltrain_i,ltrain_j)

      
      
      ltrain_ij[ltrain_ij==j] <- 0
      ltrain_ij[ltrain_ij==i] <- 1

  
       ml.logit <- glm(ltrain_ij ~ ., family = binomial(logit), data = train_ij)

       predictions <- calculateEtest(ml.logit,test,ltest)[[3]]
       predictions[predictions==1] <- i
       predictions[predictions==0] <- j
      
       k <- k+1
       
       preds_ij[k,] <- predictions
    }
       
  }
  
  preds <- vector(length = length(ltest))
  for(i in 1:length(ltest)){
      preds[i] <- names(sort(table(preds_ij[,i]),decreasing=T))[1]
  }

  errorVec <- sum(preds != ltest)/length(ltest)
    
  return(list(preds,errorVec))  
}


## ---- Obtención de valores en modelo lineal ---------------------------------------------------
P <- predictions1vs1(har.train,har.test,as.vector(lhar.train[,1]),as.vector(lhar.test[,1]))

# Matriz de confusión
Eout.linear <- P[[2]]
confussion.linear <- table(P[[1]],lhar.test[,1])

print("Etest obtenido por el modelo lineal:")
print(Eout.linear)
print("Matriz de consusión obtenida por el modelo lineal:")
print(confussion.linear)

pause()

## ---- RED NEURONAL ---------------------------------------------------
  lhar.train.multi01 <- class.ind(lhar.train[,1])
  har.data.nnet <- cbind(har.train,lhar.train.multi01)

  names(har.data.nnet) <- c(names(har.data.nnet)[1:102], "l1", "l2", "l3", "l4", "l5", "l6")
  n <- names(har.data.nnet)[1:102]

# Obtenemos la fórmula:
  f.nnet <- as.formula(paste("l1 + l2 + l3 + l4 + l5 + l6 ~",paste(n[!n %in% c("l1","l2","l3","l4","l5","l6")], collapse = " + ")))


testingLearningRate <- function(formula, data,ldata){
    rep <- sample(nrow(data), 0.3*nrow(data))
    train <- data[-rep,]
    test <- data[rep,(1:102)]
    ltest <- ldata[rep,]
    rates <- seq(from = 0.8, to = 1.2, 0.1)
    error <- vector()

    for(i in seq_along(rates)){
      ann <- neuralnet(formula = formula, data = train, hidden = 5, lifesign = 'minimal', learningrate = rates[i], rep = 3, linear.output = FALSE, act.fct = "logistic", stepmax = 10000)
      
      pr.nn <- compute(ann, test)
      pr.nn_ <- pr.nn$net.result
      pr.nn_2 <- max.col(pr.nn_)

      error[i] <- 1 - mean(pr.nn_2 == ltest)
    }
    
    return(error)
}


## ---- Estimamos los mejores valores para la tasa de aprendizaje ---------------------------------------------------
  set.seed(123456789)

  err.tasa <- testingLearningRate(f.nnet, har.data.nnet,lhar.train)

## -------------------------------------------------------
  err.tasa.rate <- cbind(c(0.8,0.9,1,1.1,1.2),  err.tasa)
  print(err.tasa.rate)
  
  pause()

## ---- Aprendemos la red neuronal y obtenemos resultados---------------------------------------------------

set.seed(996)

ann <- neuralnet(formula = f.nnet, data = har.data.nnet, hidden = 5, lifesign = 'full', learningrate = 1.1, rep = 1, linear.output = FALSE, act.fct = "logistic", stepmax = 10000)

pr.nn <- compute(ann, har.test)
pr.nn_ <- pr.nn$net.result
pr.nn_2 <- max.col(pr.nn_)
Eout.nn <- 1 - mean(pr.nn_2 == lhar.test)
confussion.nn <- table(pr.nn_2, lhar.test[,1])

print("Etest obtenido por la red neuronal:")
print(Eout.nn)
print("Matriz de confusión obtenida en la red neuronal:")
print(confussion.nn)

pause()


## ---- SVM ---------------------------------------------------
# Estimación de parámetros
tc <- tune.control(cross = 5)

svm_tune <- tune(svm, train.x = har.train, train.y = factor(lhar.train[,1]), kernel = "radial", ranges = list(gamma=c(0.01,0.1,1,10)), tunecontrol = tc)

print("Resultados de tune sobre SVM:")
print(svm_tune)

print("Errores obtenidos para cada parámetro en la validación cruzada:")
print(svm_tune$performances)

pause()

## ---- Aprendizaje del mejor modelo obtenido ---------------------------------------------------

svm_model <- svm(x = har.train, y = factor(lhar.train[,1]), kernel = "radial", gamma = 0.01)
print(summary(svm_model))

pause()

## ---- Obtenemos los resultados de la predicción ---------------------------------------------------

svm.pred <- predict(svm_model,har.test)

Eout.svm <- sum(svm.pred != lhar.test[,1])/length(svm.pred)
confussion.svm <- table(svm.pred,lhar.test[,1])

print("Etest obtenido en SVM:")
print(Eout.svm)
print("Matriz de confusión obtenida en SVM:")
print(confussion.svm)
  
pause()

## ---- BOOSTING ---------------------------------------------------
# Predicciones 1vs1
boostingPredictions1vs1 <- function(train, test, ltrain, ltest, modelNames = NULL){
  
  preds_ij <- matrix(nrow = 15, ncol = length(ltest))
  
  k <- 0
  
  for(i in 2:6){
    for(j in 1:(i-1)){
      #cat(i," vs ",j,":\n")
      
      inds_i <- which(ltrain == i)
      inds_j <- which(ltrain == j)

      ltrain_i <- ltrain[inds_i]
      ltrain_j <- ltrain[inds_j]
      
      train_i <- train[inds_i,]
      train_j <- train[inds_j,]
      
      
      train_ij <- rbind(train_i,train_j)
      ltrain_ij <- c(ltrain_i,ltrain_j)

      
      
      ltrain_ij[ltrain_ij==j] <- 0
      ltrain_ij[ltrain_ij==i] <- 1

       gbm.boost <- gbm(ltrain_ij ~ ., data = train_ij, distribution = "adaboost", n.trees = 5000)
         
         #glm(ltrain_ij ~ ., family = binomial(logit), data = train_ij)

       predictions <- predict(gbm.boost,har.test,n.trees = 5000,type = "response" )
       predictions[predictions>=0.5] <- i
       predictions[predictions<0.5] <- j
      
       k <- k+1
       
       preds_ij[k,] <- predictions
    }
       
  }
  
  preds <- vector(length = length(ltest))
  for(i in 1:length(ltest)){
      preds[i] <- names(sort(table(preds_ij[,i]),decreasing=T))[1]
  }

  errorVec <- sum(preds != ltest)/length(ltest)
    
  return(list(preds,errorVec))  
}


## ---- Obtención de resultados ---------------------------------------------------
B <- boostingPredictions1vs1(har.train,har.test,as.vector(lhar.train[,1]),as.vector(lhar.test[,1]))

# Matriz de confusión
Eout.adagbm <- B[[2]]
confussion.adagbm <- table(B[[1]],lhar.test[,1])

print("Etest obtenido en AdaBoost - gbm:")
print(Eout.adagbm)
print("Matriz de confusión obtenida en AdaBoost - gbm:")
print(confussion.adagbm)

pause()

## ---- Resultados con la generalización multiclase --------------------------------------------------
#sparsefactor -> true para aplicar regularización explicita norma L1 (?)
maboost_model <- maboost(x = har.train, y = as.factor(lhar.train[,1]), sparsefactor = TRUE)  

maboost.pred <- predict(maboost_model,har.test)

print("Etest obtenido en maboost:")
Eout.maboost <- sum(maboost.pred != lhar.test[,1])/length(maboost.pred)
print(Eout.maboost)
print("Matriz de confusión obtenida en maboost:")
confussion.maboost <- table(maboost.pred,lhar.test[,1])
print(confussion.maboost)

pause()

## ---- RANDOM FOREST-----------------------
    #La función tuneRF calcula a partir del valor por defecto de mtry el valor óptimo de mtry para el randomForest
    #Convertimos la etiqueta a un factor para que haga clasificación
    flhar.train <- as.factor(lhar.train[,1])
    best.mtry <- tuneRF(har.train, flhar.train, stepFactor = 1, improve = 0.02, ntree = 50)

## -------------------------------------------------------
  print("El valor óptimo de mtry calculado es: ")
  print(best.mtry[,1])
  
  pause()

## ----Estimación de parámetros ---------------------------------------------------
  set.seed(123456789)
   best.params <- tune(method = randomForest, har.train, flhar.train, ranges = list(ntree = c(100,200,300,400,500, 600, 700, 800, 900, 1000), mtry = 10), tunecontrol = tc)


## -------------------------------------------------------
  print(best.params$performances)

## ------------------------------------------------------
  print(best.params)
   
   pause()

## -------------------------------------------------------
  points <- best.params$performances[c(1,3)]
  plot(points, type = "l", col = "blue")
  
  pause()

## ---- Aprendizaje del mejor modelo estimado ---------------------------------------------------
set.seed(123456789)

rf_model <- randomForest(x = har.train, y = flhar.train, ntree = 500, mtry = 10)
summary(rf_model)

pause()

## ---- Obtención de resultados ---------------------------------------------------

rf.pred <- predict(rf_model,har.test)
Eout.rf <- sum(rf.pred != lhar.test[,1])/length(rf.pred)
confussion.rf <- table(rf.pred,lhar.test[,1])

print("Etest obtenido en Random Forest:")
print(Eout.rf)
print("Matriz de confusión obtenida en Random Forest:")
print(confussion.rf)
  
pause()

## ---- COMPARACIÓN GLOBAL DE RESULTADOS---------------------------------------------------

model_errors <- c(Eout.linear,Eout.nn,Eout.svm,Eout.rf,Eout.adagbm,Eout.maboost)
model_confussions <- list(confussion.linear,confussion.nn,confussion.svm,confussion.rf,confussion.adagbm,confussion.maboost)
model_names <- c("Regresión Logística","Red neuronal", "SVM", "Random Forest","Adaboost - BGM", "Adaboost - MABoost")
names(model_errors) <- model_names
names(model_confussions) <- model_names
print(model_errors)
print(model_confussions)

pause()

