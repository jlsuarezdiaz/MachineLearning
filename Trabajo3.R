## ---- echo = FALSE, warnings = FALSE, results = FALSE, message = FALSE----
    #Fijamos la semilla para obtener siempre los mismos resultados
    set.seed(123456789)

    #Añadimos librerías necesarias.
    library("caret")
    library("e1071")
    library("glmnet")
    library("leaps")


## ---- echo = FALSE, warning=FALSE, results= FALSE, message = FALSE-------
  
    #PREPARACIÓN DE LOS DATOS
    sahd <- read.table("./data/SAHD", sep=",",head=T, row.names = 1)
    summary(sahd)
 
   

## ----echo = FALSE, warning=FALSE, results = FALSE, message = FALSE-------

    #PREPROCESAMIENTO DE LOS DATOS

    #Paso 1: Modificamos las variables cualitativas (el programa no sabe bien cómo tratarlas)
    #La única variable cualitativa es famhist = {present, absent}
    sahd[,5] <- ifelse(sahd[,5]=='Present',1,0)
    colnames(sahd) <- c( 'sbp', 'tobacco', 'ldl', 'adiposity', 'present_famhist', 'typea', 'obesity', 'alcohol', 'age','chd')
    
    #Para ir explicando una a una las transformaciones
    sahd_aux <- sahd
    
    attach(sahd_aux)
    #pairs(~ sbp + tobacco + ldl + adiposity + present_famhist +typea + obesity + alcohol + age, data= sahd, col = chd+3)

## ---- echo=F,warning=F---------------------------------------------------
    #Eliminación de variables con varianza 0 o muy próximas (importante para métodos sensibles a distancias)
   
   #Ordenamos las columnas por asimetria
    sahd_asymmetry <- apply(sahd_aux, 2, skewness)
    sahd_asymmetry <- sort(abs(sahd_asymmetry), decreasing = T)
    print(sahd_asymmetry)
    readline("Pulse intro para continuar")
    
    

## ---- echo=F,warning=F---------------------------------------------------
    hist(alcohol, col = "blue")
    readline("Pulse intro para continuar")
    

## ---- echo=F,warning=F---------------------------------------------------
    BoxCoxTrans(alcohol) #No se aplica transformación pues no se ha encontrado el parámetro

## ---- echo=F,warning=F---------------------------------------------------
  correction <- 1
  c_alcohol <- alcohol + min(alcohol)+correction
  alcohol_trans <- BoxCoxTrans(c_alcohol)
  t_alcohol <- predict(alcohol_trans,c_alcohol)
  
  print("La asimetría del alcohol transformado es:")
  print(skewness(t_alcohol))
  readline("Pulse intro para continuar")
  
  
  par(mfrow = c(1,2))
  hist(alcohol, col = "red")
  hist(t_alcohol, col = "green")
  par(mfrow = c(1,1))
  readline("Pulse intro para continuar")
  
    

## ---- echo=T, warning=F--------------------------------------------------

ObjetoTrans = preProcess(sahd[,names(sahd)!="chd"],method = c("BoxCox","center","scale"))
sahdTrans <- predict(ObjetoTrans,sahd)


## ---- echo=F, warning=F--------------------------------------------------
  #Paso 5: Preparación de conjunto de train/test, y de las etiquetas train/test
  sahd.data <- sahdTrans[,-ncol(sahdTrans)]
  sahd.label <- sahdTrans[,ncol(sahdTrans)]

  train <- sample(nrow(sahd),0.7*nrow(sahd))
  sahd.train <- sahd.data[train,]
  sahd.test <- sahd.data[-train,]
  
  #Etiquetas
  lsahd.train <- sahd.label[train]
  lsahd.test <- sahd.label[-train]
  

## ---- echo=F,warning=F---------------------------------------------------
# Funciones para el cálculo del error
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
  
  return(list(ml.Etest, ml.pred_test))
}

calculateEin <- function(ml ,ltrain){
  ml.prob_train = predict(ml, type = "response") #no tenemos que introducirle el train porque recuerda
  # Ein
  ml.pred_train = rep(0, length(ml.prob_train)) #predicciones por defecto 0
  ml.pred_train[ml.prob_train >= 0.5] = 1
    
  ml.Ein = mean(ml.pred_train != ltrain)

  return(list(ml.Ein, ml.pred_train))
}

# Función para evaluar loserrores para un conjunto de modelos.
# Argumentos:
# models - lista de modelos
# test - conjunto de test
# ltrain - etiquetas de train
# ltest - etiquetas de test
# Devuelve: Matriz de num_modelos x 2. Las columnas representan (Ein, Etest)
testFamilies <- function(models, test, ltrain, ltest){

  results <- matrix(nrow=length(models), ncol = 2)
  colnames(results) <- c("Ein","Eout")
  for(i in seq_along(models)){
    results[i,1] <-calculateEin(models[[i]],ltrain)[[1]]
    results[i,2] <- 
      calculateEtest(models[[i]],test,ltest)[[1]]
  }
  
  return(results)
}

## ---- echo=F,warning=F---------------------------------------------------
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
  
  return(list(ml.Etest, ml.pred_test))
}

calculateEin <- function(ml ,ltrain){
  ml.prob_train = predict(ml, type = "response") #no tenemos que introducirle el train porque recuerda
  # Ein
  ml.pred_train = rep(0, length(ml.prob_train)) #predicciones por defecto 0
  ml.pred_train[ml.prob_train >= 0.5] = 1
    
  ml.Ein = mean(ml.pred_train != ltrain)

  return(list(ml.Ein, ml.pred_train))
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

## ---- echo=F,warning=F---------------------------------------------------

# Validación (una sola vez)
validateFamilies <- function(data,label){
  #Muestra
  train <- sample(nrow(data),0.7*nrow(data))
  
  #Datos
  data.train <- data[train,]
  data.test <-  data[-train,]
  
  #Etiquetas
  label.train <- label[train]
  label.test <- label[-train]
  
  #Modelos
  #binomial
  ml.binomial1 <- glm(chd ~ ., family = binomial(logit), data = data, subset=train)
  ml.binomial2 <- glm(chd ~ ., family = binomial(probit), data = data, subset=train)
  ml.binomial3 <- glm(chd ~ ., family = binomial(cauchit), data = data, subset=train)
  #gaussiano
  ml.gaussian1 <- glm(chd ~ ., family = gaussian(identity), data = data, subset=train)
  ml.gaussian2 <- glm(chd ~ ., family = gaussian(log), data = data, subset=train, start=rep(0, ncol(data)+1))
  #poisson
  ml.poisson1 <- glm(chd ~ ., family = poisson(log), data = data, subset=train)
  
  #quasi
  ml.quasi1 <- glm(chd ~ ., family = quasi(link = "identity", variance = "constant"), data = data, subset=train)

  #quasibinomial
  ml.quasibinomial1 <- glm(chd ~ ., family = quasibinomial(link = "logit"), data = data, subset=train)
  #quasipoisson
  ml.quasipoisson1 <- glm(chd ~ ., family = quasipoisson(link = "log"), data = data, subset=train)  

  models <- list(ml.binomial1, ml.binomial2, ml.binomial3, ml.gaussian1, ml.gaussian2, ml.poisson1, ml.quasi1, ml.quasibinomial1, ml.quasipoisson1)
  modelNames <- c("Binomial - Logit", "Binomial - Probit", "Binomial - Cauchit", "Gaussian - Identity", "Gaussian - Log", "Poisson", "Quasi", "Quasibinomial","Quasipoisson")
  
  testFamilies(models, data.test, label.train, label.test, modelNames)
  
}

# Función para realizar multiples validaciones
repValidation <- function(rep,data,label){
  l <- replicate(n = rep, expr = validateFamilies(data,label))
  #l es una matriz 3D de rep x num_modelos x 2 (Ein,Eout)
  #l[,,i] -> experimento i-ésimo
  #[,i,] -> Ein / Eout de cada modelo en los distintos experimentos (i=1 Ein, i=2 Eout)
  #[i,,] -> Ein y Eout para el modelo i-ésimo en cada experimento
  apply(FUN = mean, X = l, MARGIN = c(1,2))
}


## ---- echo=F, warning=F--------------------------------------------------
# Comprobamos qué método proporciona un menor error de validación con múltiples validaciones
repValidation(rep = 100, sahd.data, sahd.label)


# Confirmamos que el modelo que mejor generaliza es el la gaussiana2
ml.sahd <- glm(chd ~ ., family = poisson(log), data = sahd.data, subset=train, start=rep(0, ncol(sahd.data)+1))



## ---- echo=F,warning=F---------------------------------------------------
  #library(glmnet)
  #El método para llamar a GLM con lasso (elasticnet regularization) es glmnet

  #En la documentación aqui parece que haya más families.

  #Mediante validación cruzada sacamos el mejor lambda

  testRegularization <- function(data, label, iter = 100){
      error1 <- 0
      error2 <- 0
      error3 <- 0
      
      for(i in 1:iter){
          train <- sample(nrow(data),0.7*nrow(data))
          data.train <- data[train,]
          data.test <- data[-train,]
          #Etiquetas
          ldata.train <- label[train]
          ldata.test <- label[-train]
          
          #Ponemos el link en las etiquetas pues a glmnet no se le puede pasar como argumento
          ml_lasso <- cv.glmnet(as.matrix(data.train), ldata.train, family="poisson")
          ml = ml.gaussian2 <- glm(chd ~ ., family = gaussian(log), data = data, subset=train, start=rep(0, ncol(data)+1))
    
          error1 <- error1 + calculateEtest(ml_lasso,data.test,ldata.test,ml_lasso$lambda.min)[[1]]
          error2 <- error2 +calculateEtest(ml_lasso,data.test,ldata.test,ml_lasso$lambda.1se)[[1]]
          error3 <- error3 +calculateEtest(ml,data.test,ldata.test)[[1]]
      }
    
      #print(error1/100)
      #print(error2/100)
      #print(error3/100)
      return(c(error1/iter,error2/iter,error3/iter))
  }

  l <- testRegularization(sahd.data, sahd.label, 100)

  cat("Etest con lambda min: ",l[1],"\n")
  readline("Pulse intro para continuar")

  cat("Etest con lambda 1se: ",l[2],"\n")
  readline("Pulse intro para continuar")
  
  cat("Etest con el modelo original: ",l[3],"\n")
  readline("Pulse intro para continuar")
  


## ---- echo=F,warning=F---------------------------------------------------
  ml_lasso <- cv.glmnet(as.matrix(sahd.train), lsahd.train, family="poisson")
  coeffs.1se <- coef(ml_lasso, s = ml_lasso$lambda.1se)
  print(coeffs.1se)
  readline("Pulse intro para continuar")
  

## ---- echo=F,warning=F---------------------------------------------------

  #method = "exhaustive" para que no sea greedy.
  testing <- regsubsets(chd ~ ., data = sahd.data, nbest = 1, method = "exhaustive")

  #Obtenemos una tablita con los que es mejor coger.
  stesting <- summary(testing)
  print(stesting$outmat)
  readline("Pulse intro para continuar")
  

## ---- echo=F,warning=F---------------------------------------------------

  #Dibujar -> testing$ress
  plot(testing$ress, col = "red", pch = 19, type = "b")
  readline("Pulse intro para continuar")
  

## ---- echo=F,warning=F---------------------------------------------------

  plot(stesting$bic, col = "blue", pch = 19, type = "b")
  readline("Pulse intro para continuar")
  

## ---- echo=F,warning=F---------------------------------------------------

  sahd.data <- sahd.data[,c(2,3,5,6,9)]

## ---- echo=F,warning=F---------------------------------------------------
testPolyTransform <- function(data,label){
  #Muestra
  train <- sample(nrow(data),0.7*nrow(data))
  
  #Datos
  data.train <- data[train,]
  data.test <-  data[-train,]
  
  #Etiquetas
  label.train <- label[train]
  label.test <- label[-train]
  
  #Usamos el modelo ganador
  ml <- glm(chd ~., family = gaussian(log), data = data, subset =train,start=rep(0, ncol(data)+1))
  
  models <- list(ml)
  
  testFamilies(models, data.test, label.train, label.test)
  
}

testPolyTransformRep <- function(rep,data,label){
   l <- replicate(n = rep, expr = testPolyTransform(data,label))
  #l es una matriz 3D de rep x num_modelos x 2 (Ein,Eout)
  #l[,,i] -> experimento i-ésimo
  #[,i,] -> Ein / Eout de cada modelo en los distintos experimentos (i=1 Ein, i=2 Eout)
  #[i,,] -> Ein y Eout para el modelo i-ésimo en cada experimento
  apply(FUN = mean, X = l, MARGIN = c(1,2))
}

# Devuelve un vector de coeficientes para cada atributo
# tol: Nivel de tolerancia: si dos Eout se diferencian en menos de Eout, escogemos el coeficiente más simple
polynomialTransformGreedy <- function(data,label,max_coeff, tol = 0){
  transform <- data
  coeffs <- c()

  for(i in 1:ncol(transform)){
    bestEout <- 1
    bestInd <- 0
    for(j in 1:max_coeff){
      transform[,i] <- I(data[,i]^j)
      v <- testPolyTransformRep(100,transform,label)
      # Comparamos con Etest
      if(v[2] + tol < bestEout){
        bestEout <- v[2]
        bestInd <- j
      }
      cat("Attr = ",i,", Exp = ",j,", ","Ein = ", v[1], "Eout = ",v[2],"\n")
    }
    #Nos quedamos con el mejor coeficiente obtenido para el atributo
    transform[,i] <- I(data[,i]^bestInd)
    coeffs <- c(coeffs,bestInd)
  }
  # Por ser greedy con el mecanismo escogido el mejor Eout va a ser el de la última iteración
  list(coeffs,bestEout)
}

## ---- echo=F,warning=F---------------------------------------------------
l <- polynomialTransformGreedy(sahd.data,sahd.label,6, tol = 0.01)
cat("Vector de exponentes: ", l[[1]])
readline("Pulse intro para continuar")

cat("Eout estimado: ",l[[2]])
readline("Pulse intro para continuar")



## ---- echo=F,warning=F---------------------------------------------------
pairs( ~ tobacco + ldl + present_famhist + typea + age, data=sahd.data, col = chd+3)
readline("Pulse intro para continuar")


## ---- echo=F,warning=F---------------------------------------------------

ml <- glm(chd ~ typea+age , family = poisson(log), data = sahd.data, subset=train)


ml.probtrain <- predict(ml,type="response")
ml.predtrain <- rep(0,length(ml.probtrain))
ml.predtrain[ml.probtrain >= 0.5]=1
ml.Ein = mean(ml.predtrain != lsahd.train)
  
ml.probtest <- predict(ml,sahd.test,type="response")
ml.predtest <- rep(0,length(ml.probtest))
ml.predtest[ml.probtest >= 0.5]=1
ml.Etest = mean(ml.predtest != lsahd.test)

cat("Ein = ",ml.Ein,"\n")
readline("Pulse intro para continuar")

cat("Etest = ",ml.Etest,"\n")
readline("Pulse intro para continuar")


coefs_recta_explicita <- function(coefs_recta_impl){
  return(c(-coefs_recta_impl[2]/coefs_recta_impl[3],-coefs_recta_impl[1]/coefs_recta_impl[3]))
}



## ----echo = FALSE, warning=FALSE-----------------------------------------
    #PREPARACIÓN DE LOS DATOS
    ozone.df <- read.table("./data/ozone", sep=",",head=T)
    summary(ozone.df)

    attach(ozone.df)


## ---- echo=F,warning=F---------------------------------------------------

    pairs( ~ozone + vh + wind + humidity + temp + ibh +dpg + ibt + vis + doy, data= ozone.df, col = "red")
    readline("Pulse intro para continuar")
    

## ---- echo=F, warning=F--------------------------------------------------

#library(caret)

#No aplicamos PCA pues pierde variabilidad.
ObjetoTrans2 = preProcess(ozone.df[,names(ozone.df)!="ozone"],method = c("BoxCox","center","scale"))

ozoneTrans <- predict(ObjetoTrans2,ozone.df)


## ----echo=F,warning=F----------------------------------------------------

attach(ozoneTrans)
pairs( ~ozone + vh + wind + humidity + temp + ibh +dpg + ibt + vis + doy, data= ozoneTrans, col = "blue")
readline("Pulse intro para continuar")


## ---- echo=F, warning=F--------------------------------------------------
  #Paso 5: Preparación de conjunto de train/test, y de las etiquetas train/test
  ozone.data <- ozoneTrans[,-1]
  ozone.label <- ozoneTrans[,1]
  #sahd.label[sahd.label==0] <- -1 
  
  train <- sample(nrow(ozone.df),0.7*nrow(ozone.df))
  ozone.train <- ozone.data[train,]
  ozone.test <- ozone.data[-train,]
  
  #Etiquetas
  lozone.train <- ozone.label[train]
  lozone.test <- ozone.label[-train]
  

## ---- echo=F,warning=F---------------------------------------------------
calculateEtestRegression <- function(mr, test, ltest, s = 0){
    if(s==0){
      mr.pred_test = predict(mr, test, type="response")
    }else{
      mr.pred_test = predict(mr,as.matrix(test),type = "response", s = s)
    }
    #mr.pred_test = predict(mr, test, type="response")
    mr.Etest <- mean((mr.pred_test-ltest)^2)
    
    return(list(mr.Etest,mr.pred_test))
}

calculateEinRegression <- function(mr ,ltrain){
    mr.pred_train = predict(mr,type = "response")
    mr.Ein <- mean((mr.pred_train-ltrain)^2)

    return(list(mr.Ein, mr.pred_train))
}

testFamiliesRegression <- function(models, test, ltrain, ltest){

  results <- matrix(nrow=length(models), ncol = 2)
  colnames(results) <- c("Ein","Eout")
  for(i in seq_along(models)){
    results[i,1] <-calculateEinRegression(models[[i]],ltrain)[[1]]
    results[i,2] <- 
      calculateEtestRegression(models[[i]],test,ltest)[[1]]
  }
  
  return(results)
}

## ---- echo=F,warning=F---------------------------------------------------

# Validación (una sola vez)
validateFamiliesRegression <- function(data,label){
  #Muestra
  train <- sample(nrow(data),0.7*nrow(data))
  
  #Datos
  data.train <- data[train,]
  data.test <-  data[-train,]
  
  #Etiquetas
  label.train <- label[train]
  label.test <- label[-train]
  
  #Modelos
  mr1 = lm(ozone ~ ., data, train)


  models <- list(mr1)
  
  testFamiliesRegression(models, data.test, label.train, label.test)
  
}

# Función para realizar multiples validaciones
repValidationRegression <- function(rep,data,label){
  l <- replicate(n = rep, expr = validateFamiliesRegression(data,label))
  #l es una matriz 3D de rep x num_modelos x 2 (Ein,Eout)
  #l[,,i] -> experimento i-ésimo
  #[,i,] -> Ein / Eout de cada modelo en los distintos experimentos (i=1 Ein, i=2 Eout)
  #[i,,] -> Ein y Eout para el modelo i-ésimo en cada experimento
  apply(FUN = mean, X = l, MARGIN = c(1,2))
}


## ---- echo=F,warning=F---------------------------------------------------
  repValidationRegression(100,ozone.data,ozone.label)
  readline("Pulse intro para continuar")


## ---- echo=F,warning=F---------------------------------------------------

testRegularizationRegression <- function(data, label, iter = 100){
      error1 <- 0
      error2 <- 0
      error3 <- 0
      
      for(i in 1:iter){
          train <- sample(nrow(data),0.7*nrow(data))
          data.train <- data[train,]
          data.test <- data[-train,]
          #Etiquetas
          ldata.train <- label[train]
          ldata.test <- label[-train]
          
          #Ponemos el link en las etiquetas pues a glmnet no se le puede pasar como argumento
          #grid=10^seq(10,-2,length=100)
          ml_lasso <- cv.glmnet(as.matrix(data.train), ldata.train)
          ml = lm(ozone ~ ., data, train)
    
          error1 <- error1 + calculateEtestRegression(ml_lasso,data.test,ldata.test,ml_lasso$lambda.min)[[1]]
          error2 <- error2 +calculateEtestRegression(ml_lasso,data.test,ldata.test,ml_lasso$lambda.1se)[[1]]
          error3 <- error3 +calculateEtestRegression(ml,data.test,ldata.test)[[1]]
      }
    
      return(c(error1/iter,error2/iter,error3/iter))
  }

## ---- echo=F,warning=F---------------------------------------------------
  #library(glmnet)


  l <- testRegularizationRegression(ozone.data, ozone.label, 100)

  cat("Etest con lambda min: ",l[1],"\n")
  readline("Pulse intro para continuar")
  
  cat("Etest con lambda 1se: ",l[2],"\n")
  readline("Pulse intro para continuar")
  
  cat("Etest con el modelo original: ",l[3],"\n")
  readline("Pulse intro para continuar")
  

## ---- echo=F,warning=F---------------------------------------------------
  #method = "exhaustive" para que no sea greedy.
  testing <- regsubsets(ozone ~ ., data = ozone.data, nbest = 1, method = "exhaustive")

  #Obtenemos una tablita con los que es mejor coger.
  stesting <- summary(testing)
  print(stesting$outmat)
  readline("Pulse intro para continuar")
  

## ---- echo=F,warning=F---------------------------------------------------
  #Dibujar -> testing$ress
  plot(testing$ress, col = "red", pch = 19, type = "b")
  readline("Pulse intro para continuar")
  

## ---- echo=F,warning=F---------------------------------------------------

  plot(stesting$bic, col = "blue", pch=19, type = "b")
  readline("Pulse intro para continuar")
  
  ozone.data_regu <- ozone.data[c(3,4,7,9)]


## ---- echo=F,warning=F---------------------------------------------------
testPolyTransformRegression <- function(data,label){
  #Muestra
  train <- sample(nrow(data),0.7*nrow(data))
  
  #Datos
  data.train <- data[train,]
  data.test <-  data[-train,]
  
  #Etiquetas
  label.train <- label[train]
  label.test <- label[-train]
  
  #Usamos el modelo ganador
  mr <- lm(ozone ~ ., data, train)
  
  models <- list(mr)
  
  testFamiliesRegression(models, data.test, label.train, label.test)
  
}

testPolyTransformRepReg <- function(rep,data,label){
   l <- replicate(n = rep, expr = testPolyTransformRegression(data,label))
  #l es una matriz 3D de rep x num_modelos x 2 (Ein,Eout)
  #l[,,i] -> experimento i-ésimo
  #[,i,] -> Ein / Eout de cada modelo en los distintos experimentos (i=1 Ein, i=2 Eout)
  #[i,,] -> Ein y Eout para el modelo i-ésimo en cada experimento
  apply(FUN = mean, X = l, MARGIN = c(1,2))
}

# Devuelve un vector de coeficientes para cada atributo
# tol: Nivel de tolerancia: si dos Eout se diferencian en menos de Eout, escogemos el coeficiente más simple
polynomialTransformGreedyRegression <- function(data,label,max_coeff, tol = 0){
  transform <- data
  coeffs <- c()

  for(i in 1:ncol(transform)){
    bestEout <- Inf
    bestInd <- 0
    for(j in 1:max_coeff){
      transform[,i] <- I(data[,i]^j)
      v <- testPolyTransformRepReg(100,transform,label)
      # Comparamos con Etest
      if(v[2] + tol < bestEout){
        bestEout <- v[2]
        bestInd <- j
      }
      cat("Attr = ",i,", Exp = ",j,"Ein = ",v[1], ", Eout = ",v[2],"\n")
    }
    #Nos quedamos con el mejor coeficiente obtenido para el atributo
    transform[,i] <- I(data[,i]^bestInd)
    coeffs <- c(coeffs,bestInd)
  }
  # Por ser greedy con el mecanismo escogido el mejor Eout va a ser el de la última iteración
  list(coeffs,bestEout)
}

## ---- echo=F,warning=F---------------------------------------------------

l <- polynomialTransformGreedyRegression(ozone.data_regu,ozone.label,4, tol = 0.01)

cat("Vector de exponentes: ", l[[1]],"\n")
readline("Pulse intro para continuar")

cat("Eout estimado: ",l[[2]],"\n")
readline("Pulse intro para continuar")

## ---- echo=F,warning=F---------------------------------------------------

l <- polynomialTransformGreedyRegression(ozone.data,ozone.label,4, tol = 0.01)

cat("Vector de exponentes: ", l[[1]],"\n")
readline("Pulse intro para continuar")

cat("Eout estimado: ",l[[2]],"\n")
readline("Pulse intro para continuar")



