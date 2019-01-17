#AUX functions fit

#------------------ DUMMY VARS ENCODINNG -----------------------
dummy_vars_encoding <- function(df){
  string_columns= !sapply(df, is.numeric)
  string_data_set = df[,!sapply(df, is.numeric), drop=FALSE ]
  string_data_set_columns = names(string_data_set)
  
  for (column in string_data_set) {
    new_columns= levels(column)
    for (new_column in new_columns){
      df[[paste("dummy_", new_column)]] = (column == new_column)*1
      
    }
  }
  
  for(name in string_data_set_columns){
    df[[name]] = NULL
  }
  
  return(df)
}



apply_dummy_encoding_to_test<-function(X_train_encoded, X_test){
  X_test_encoded = dummy_vars_encoding(X_test)
  missing_x_test_columns = names(X_train_encoded)[!names(X_train_encoded)%in%names(X_test_encoded)]
  missing_x_train_columns = names(X_test_encoded)[!names(X_test_encoded)%in%names(X_train_encoded)]
  for (i in missing_x_test_columns){X_test_encoded[[i]]=0}
  for (i in missing_x_train_columns){X_train_encoded[[i]]=0}
  
  return(X_test_encoded, X_train_encoded) 
  #R does not support multiple returns in fuctions. The best way to do it is to return multiple objects in an array
  #the problem is that it will distorce the information if the objects to be returned are list
}


#------------------ NA HANDLING -----------------------

remove_nan_rows <- function(X_train){
  rows_to_keep = !apply(object@X_train, 1, function(row) any(is.na(row)))
  X_train=X_train[rows_to_keep,]
  return(X_train)
}


knn_na = function(df, k, distance, tie_break){
  if(sum(is.na(df))==0){
    return(df)
  }else{
    base_data = df[apply(df,1, function(row) !is.na(sum(row))),]
    base_y = df[apply(df,1, function(row) !is.na(sum(row))),]
    rows_to_fill = !apply(df,1, function(row) !is.na(sum(row)))
    n_rows_to_fill = sum(rows_to_fill)
    data_to_fill = df[rows_to_fill,]
    for(i in seq(n_rows_to_fill)){
      aux = data_to_fill[i,]
      total_aux_na = sum(is.na(aux))
      for(j in seq(total_aux_na)){
        vars_to_predict = names(aux)[is.na(aux)]
        sub_aux = aux[,!names(aux)%in%vars_to_predict]
        sub_base_data = base_data[,!names(base_data)%in%vars_to_predict]
        sub_y =  base_data[[vars_to_predict[1]]]
        indexes = knn_search(sub_base_data, sub_aux, k, distance)
        if(grepl('dummy',vars_to_predict[1],fixed=T)){
          aux[[vars_to_predict[1]]] = round(knn_output_selector(distance, indexes,sub_y, tie_break = tie_break))
        }else{
          aux[[vars_to_predict[1]]] = knn_output_selector('regression', indexes,sub_y, tie_break = tie_break)
        }
      }
      base_data[nrow(base_data)+1,]=aux
    }
    return(base_data)
  }
}



####--------------- AUX FUNCTIONS TO GET PREDICTION  --------------------------------

knn_search <- function(X_train, X_test, k, distance){
  if (distance=='manhattan'){
    differences <- as.matrix(t(apply(as.matrix(X_train),1,function(row) abs(row-as.matrix(X_test)))))
    differences <- if(nrow(differences) == 1) t(differences) else differences
    distance <- apply(differences, 1, sum)
  } else if (distance=='euclidean'){
    differences <- as.matrix(t(apply(as.matrix(X_train),1,function(row) (row-as.matrix(X_test))^2)))
    differences <- if(nrow(differences) == 1) t(differences) else differences
    distance <- sqrt(apply(differences, 1, sum))
  }
  knn_ix= sort(distance, index.return=T)$ix[1:k]
  return(knn_ix)
}


knn_output_selector <- function(model_type, indexes, y, tie_break){
  if (model_type=='classification') {
    if (tie_break=='reduce_k'){
      return(get_class_recursive_reduce_k(y, indexes))
    } else if(tie_break=='random'){
      return(get_class_random(y, indexes))
    } else {
      return('error - unexistant tie break critera')
    }
    
  } else {
    return(get_regression_with_mean(y,indexes))
  }
}


get_class_recursive_reduce_k <- function(y, indexes){
  neighbours = y[indexes]
  frequencies = table(neighbours)
  neighbours_with_max_freq <- frequencies[frequencies==max(frequencies)]
  if (length(neighbours_with_max_freq)>1){
    return(get_class_recursive_reduce_k(y,indexes[-length(indexes)]))
  }else{
    return(names(neighbours_with_max_freq))
  }
}


get_class_random <- function(y, indexes){
  neighbours = y[indexes]
  frequencies = table(neighbours)
  neighbours_with_max_freq <- frequencies[frequencies==max(frequencies)]
  return(sample(names(neighbours_with_max_freq),1))
}

get_regression_with_mean <- function(y, indexes){
  neighbours = y[indexes]
  return(mean(neighbours))
}


# define an S4 class for a knn with default vaklues
setClass("knn", 
         representation(distance="character", k= "numeric",
                               tie_break='character',na_handling='character',
                               X_train="data.frame", model_type='character',
                               y_train="factor"),
         prototype(distance = "euclidean", k = 3, tie_break='reduce_k',
                   na_handling='remove NA'))

# create a generic method called 'distane' that dispatches
# on the type of object it's applied to
setGeneric(
  "distance",
  function(object) {
    standardGeneric("distance")
  }
)

setMethod(
  "distance",
  signature("knn"),
  function(object) {
    paste("The distance metric is", paste(object@distance, collapse=", "))
  }
)

setGeneric(
  "model_type",
  function(object) {
    standardGeneric("model_type")
  }
)

setMethod(
  "model_type",
  signature("knn"),
  function(object) {
    if (length(object@model_type)==0){
      paste("No model type. Fit the model first using fit().")
    }else{paste("The model type  is", paste(object@model_type, collapse=", "))}
      
  
    
  }
)

setGeneric(
  "fit",
  function(object, X_train, y_train) {
    standardGeneric("fit")
  }
)

setMethod(
  "fit",
  signature("knn"),
  function(object, X_train, y_train) {
    
    #define model type - regression or classification
    object@model_type = if(is.numeric(object@y_train)) "regression" else "classification"
    
    object@y_train=y_train
    
    #storage train data with binary encoding
    object@X_train= dummy_vars_encoding(X_train)
    
    
    #handle NA values
    if(object@na_handling=='remove NA'){
      
      rows_to_keep = !apply(object@X_train, 1, function(row) any(is.na(row)))
      object@X_train = object@X_train[rows_to_keep,]
      object@y_train=y_train[rows_to_keep]
      
      
    } else{
      
      object@X_train = knn_na(object@X_train, k=object@k, distance=object@distance, tie_break = object@tie_break)
    }
    
    
    return(object)
  }
)

setGeneric(
  "predict",
  function(object, X_test) {
    standardGeneric("predict")
  }
)

setMethod(
  "predict",
  signature("knn"),
  function(object, X_test) {
    "apply dummy encoding to test set and add new dummy collumns to either train and
    test data in case of new attribute values for that sets"
    
    X_test = dummy_vars_encoding(X_test)
    missing_x_test_columns = names(object@X_train)[!names(object@X_train)%in%names(X_test)]
    missing_x_train_columns = names(X_test)[!names(X_test)%in%names(object@X_train)]
    for (i in missing_x_test_columns){X_test[[i]]=0}
    for (i in missing_x_train_columns){object@X_train[[i]]=0}
    
    "Find X_test nearest neighbours"
    k_neighbours = t(apply(X_test, 1, function(row) knn_search(X_train = object@X_train,
                                                               row,k=object@k, distance = object@distance)))
    #return class
    
    y_test = apply(k_neighbours, 1, function(row) knn_output_selector(object@model_type, row, y_train, object@tie_break))
    
    return(y_test)
  }
)




#import data
df = read.csv('./Documents/MADSAD/Programming and Databases/dataset3.csv')
X_train= df[ , (names(df)  != 'credito')]
X_train
y_train = df[ , (names(df)  = 'credito')]
y_train
x_test =  data.frame(age=c(10,15,16), Height=c(1.2,1.6,1.7), state=c('B','A','C'))


# create some knn
k_nn <- new("knn", k=3, distance="euclidean", tie_break='reduce_k', na_handling='knn_na')

#get model type
model_type(k_nn)

#train some knn
k_nn<-fit(k_nn, X_train=X_train, y_train = y_train)

#get model type
model_type(k_nn)

#get knn predictions
y_test<-predict(k_nn, x_test)

y_test



