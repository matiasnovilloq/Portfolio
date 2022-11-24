rm(list=ls())

library(xgboost)
library(data.table)
library(tm)
library(Matrix)
library(dplyr)
library(ggplot2)
library(clue)

setwd("C:/Users/Matias Novillo/Desktop/MIM/Data Mining/TP/")

#Importamos las funciones que tenemos en un script aparte
source("functions.R")

##~~~~~~~~~~~~~ Cagamos los datos ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DATA_PATH <- "C:/Users/Matias Novillo/Desktop/MIM/Data Mining/TP/competition_data_v2/competition_data/"

#Cargamos una muestra del 50% de los datos
ads_data <- load_competition_data(DATA_PATH, sample_ratio = 1,from_when = "2021_07")
#ads_data <- readRDS("ads_data_50pct.RDS")
#ads_data <- readRDS("ads_data_2021_07.RDS")

#Eliminamos los datos de fin de septiembre (en verdad no son 0)
ads_data <- ads_data[!((strftime(ads_data$created_on, "%Y-%m", tz="UTC") == "2021-09") & (strftime(ads_data$created_on, "%d", tz="UTC") >= 17)),]

##~~~~~~~~~~~~~ Guardamos una variable que identifique training, validation y testing ~~~~~~~~~~~~~
ads_data$train_val_eval <- ifelse(ads_data$created_on >= strptime("2021-10-01", format = "%Y-%m-%d", tz = "UTC"), "eval", "train")
ads_data[sample(which(ads_data$train_val_eval == "train"), round(0.05 * sum(ads_data$train_val_eval == "train"))), "train_val_eval"] <- "valid"

View(head(ads_data))

##~~~~~~~~~~~~~ Hacemos feature engineering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Creamos variables a partir de la fecha para poder captar comportamientos estilo estacionalidades
ads_data$day <- as.integer(strftime(ads_data$created_on, format = "%d", tz = "UTC")) #Dia
ads_data$month <- as.integer(strftime(ads_data$created_on, format = "%m", tz = "UTC")) #Mes
ads_data$year <- as.integer(strftime(ads_data$created_on, format = "%Y", tz = "UTC")) #Anio
ads_data$week_day <- as.integer(strftime(ads_data$created_on, format = "%w", tz = "UTC")) #Dia de la semana
ads_data$year_week <- as.integer(strftime(ads_data$created_on, format = "%W", tz = "UTC")) #Semana del anio

#Analizamos la variabilidad de las variables
mean(is.na(ads_data$place_l6)) #place_l6 tiene 97% de NA
mean(is.na(ads_data$place_l5)) #place_l5 tiene 87% de NA
unique(ads_data$place_l6)
unique(ads_data$place_l5)

ads_data$place_l6 <- ifelse(is.na(ads_data$place_l6), 0, 1)
ads_data$place_l5 <- ifelse(is.na(ads_data$place_l5), 0, 1)

#Agregamos combinaciones de variables y ratios
ads_data <- ads_data %>% mutate(bed_and_bath = bathrooms + bedrooms,
                                rooms_bed = rooms/bedrooms,
                                rooms_bath = rooms/bathrooms,
                                bed_bath = bedrooms/bathrooms,
                                surfacecovered_surfacetotal = surface_covered/surface_total,
                                surfacecovered_rooms = surface_covered/rooms,
                                surfaceuncovered =  surface_total-surface_covered,
                                price_m2_covered = price_usd/surface_covered,
                                price_m2 = price_usd/surface_total,
                                price_m2_uncovered = price_usd/surfaceuncovered,
                                price_bath = price_usd/bathrooms,
                                price_bedrooms = price_usd/bedrooms,
                                price_bed_and_bath = price_usd/bed_and_bath,
                                price_rooms = price_usd/rooms)

ratios = c("rooms_bed", "rooms_bath", "bed_bath", "surfacecovered_surfacetotal", "surfacecovered_rooms",
           "surfaceuncovered","price_m2_covered", "price_m2","price_m2_uncovered","price_bath",
           "price_bedrooms","price_bed_and_bath","price_rooms")

#Algunos denomidadores son 0, corrijo Inf
ads_data[, ratios] <- lapply(ads_data[, ..ratios], function(x) ifelse(is.infinite(x), NA, x))

dplyr::count(ads_data[(is.na(ads_data$bed_bath))], property_type, sort = TRUE)
#Hay muchos departamentos y casas que no tienen informacion de bedrooms y/o bathrooms. Se les ocurre forma para tratar de conseguirlo?

# Aquí sería un buen lugar para crear nuevas variables
ads_data$surface_by_room <- ads_data$surface_total / ads_data$rooms
ads_data$train_sample <- ads_data$created_on < strptime("2021-10-01", format = "%Y-%m-%d", tz = "UTC")
ads_data$created_on <- as.numeric(ads_data$created_on)

# MG - NUEVAS VARIABLES: 

# QUEREMOS SOLUCIONAR EL PROBLEMA DE LOS NA EN LA COLUMNA ROOMS, CON EL OBJETIVO DE MEJORAR EL DESEMPEÑO DEL MODELO.

# PASO 1: CREAMOS UNA NUEVA COLUMNA QUE CALCULA LA MEDIA DE LA DIFERENCIA ENTRE ROOMS Y BEDROOMS (SIN CONTAR LOS NA), ESTO LO VAMOS A USAR MÁS ADELANTE. 

ads_data$media_diferencia_ambientes <- as.numeric((ads_data$rooms - ads_data$bedrooms), na.rm=TRUE) # DIFERENCIA ENTRE ROOMS Y BEDROOMS
ads_data$media_diferencia_ambientes <- (ads_data$media_diferencia_ambientes **2)** (1/2) # HACEMOS POSITIVOS LOS DATOS (POTENCIA DE DOS Y RAIZ CUADRADA)
ads_data$media_diferencia_ambientes <- round(mean(ads_data$media_diferencia_ambientes)) # CALCULAMOS LA MEDIA ENTERA

# PASO 2: VAMOS A CREAR UNA NUEVA COLUMNA LLAMADA "AMBIENTES" PARA IDENTIFICAR LOS AMBIENTES DEL INMUEBLE.
# LA COLUMNA AMBIENTES, EN LOS CASOS EN QUE HAYA DATO DE "ROOMS", VA A MANTENER ESE DATO. CUANDO ROOMS SEA NA, VA A TOMAR EL VALOR DE BEROOMS + 1 (DADO QUE LA MEDIA DE DIFERENCIA ENTRE ROOMS Y BEDROOMS ERA 1). CUANDO ROOMS Y BEDROOMS SEAN NA, EL VALOR QUE VA A TOMAR LA COLUMNA AMBIENTES, ES LA MEDIA DE LOS CUARTOS.

ads_data$cuartos_ambientes <- ads_data$bedrooms + ads_data$media_diferencia_ambientes # SUMAMOS LAS BEDROOMS MÁS LA MEDIA DE LA DIFERENCIA QUE CALCULAMOS RECIEN
ads_data$ambientes <- ifelse(!is.na(ads_data$rooms), ads_data$rooms, ads_data$cuartos_ambientes) # EN AMBIENTES, SI NO ES NA, QUE ME TIRE LOS ROOMS, SI ES NO, BEDROOMS MAS LA DIFERENCIA
#ads_data$media_ambientes <- round(mean(ads_data$rooms, na.rm=TRUE))  # LA MEDIA DE LOS ROOMS
#ads_data$ambientes <- ifelse(!is.na(ads_data$ambientes), ads_data$ambientes, ads_data$media_ambientes) #SI AMBIENTES ES NA, QUE ME PONGA LA MEDIA DE LOS ROOMS.



#Podemos aprovechar el texto!
ads_data$pileta <- grepl(paste(c("pileta", "piscina"), collapse = "|"), tolower(ads_data$description))
ads_data$luminoso <- grepl("luminos", tolower(ads_data$description))
ads_data$vista <- grepl("vista", tolower(ads_data$description))
ads_data$jardin <- grepl("jardin", chartr("ÁÉÍÓÚ", "AEIOU", tolower(ads_data$description)))
ads_data$luz <- grepl("luz", tolower(ads_data$description))
ads_data$moderno <- grepl("moderno", tolower(ads_data$description))
ads_data$amoblado <- grepl(paste(c("amoblad", "amueblad"), collapse = "|"), tolower(ads_data$description))
ads_data$cochera <- grepl(paste(c("cochera", "garage", "estacionamiento", "parqueadero", "parking"), collapse = "|"), tolower(ads_data$description))
ads_data$balcon <- grepl("balcon", tolower(ads_data$description))
ads_data$terraza <- grepl("terraza", tolower(ads_data$description))

ads_data$pileta <- grepl(paste(c("pileta", "piscina"), collapse = "|"), tolower(ads_data$title))
ads_data$luminoso <- grepl("luminos", tolower(ads_data$title))
ads_data$vista <- grepl("vista", tolower(ads_data$title))
ads_data$jardin <- grepl("jardin", chartr("ÁÉÍÓÚ", "AEIOU", tolower(ads_data$title)))
ads_data$luz <- grepl("luz", tolower(ads_data$title))
ads_data$moderno <- grepl("moderno", tolower(ads_data$title))
ads_data$amoblado <- grepl(paste(c("amoblad", "amueblad"), collapse = "|"), tolower(ads_data$title))
ads_data$cochera <- grepl(paste(c("cochera", "garage", "estacionamiento", "parqueadero", "parking"), collapse = "|"), tolower(ads_data$title))
ads_data$balcon <- grepl("balcon", tolower(ads_data$title))
ads_data$terraza <- grepl("terraza", tolower(ads_data$title))



View(tail(ads_data))

#Podemos utilizar fuentes externas!
mobility <- readRDS("mobility.RDS")
mobility$country_region <- ifelse(mobility$country_region == "Peru", "Perú", mobility$country_region)
mobility$country_region <- as.factor(mobility$country_region)
mobility$date <- strftime(mobility$date, "%Y-%m-%d", tz="UTC")
mobility <- as.data.table(mobility)

ads_data$date <- strftime(ads_data$created_on, "%Y-%m-%d", tz="UTC")
ads_data <- merge(ads_data,mobility, by.x = c("place_l1", "date"), by.y = c("country_region", "date"), all.x = TRUE)

#Probemos crear variables utilizando aprendizaje no supervisado
train_data <- ads_data[ads_data$train_val_eval == "train"]
kmeans_columns_to_keep <- c('price_usd', 'rooms', 'bathrooms', 'bedrooms', 'bed_and_bath','lat','lon',ratios)
train_data_clusters <- train_data[, ..kmeans_columns_to_keep]
train_data_clusters <- train_data_clusters[complete.cases(train_data_clusters)]

#Escalamos y eliminamos outliers
train_data_clusters <- as.data.frame(sapply(train_data_clusters, function(data) (abs(data-mean(data))/sd(data))))    
train_data_clusters <- train_data_clusters[!rowSums(train_data_clusters>4),]

kmeans_results <- find_k_means(train_data_clusters,3,20)
#kmeans_results <- readRDS("kmeans_result.RDS")

plot(c(1:18), kmeans_results$var, type="o",
     xlab="# Clusters", ylab="tot.withinss")

#Entrenamos KMeans
clusters_model <- kmeans(train_data_clusters,
                   centers=7, iter.max=3000, nstart=10)

#Hay diferencias entre grupos?
train_data$cluster <- factor(cl_predict(clusters_model, train_data[, ..kmeans_columns_to_keep]))
saveRDS(train_data,"train_data_clusters.RDS")
train_data <- readRDS("train_data_clusters.RDS")

train_data %>% group_by(cluster) %>% summarise(prom_contactos = mean(contacts, na.rm=TRUE),
                                  cant_obs = n(),
                                  cant_obs_pct = n()/nrow(.))

#Predecimos el cluster para todo el dataframe
ads_data$cluster <- factor(cl_predict(clusters_model, ads_data[, ..kmeans_columns_to_keep]))

#La fecha entera sirve? Como la usamos?
ads_data$created_on <- as.numeric(ads_data$created_on)

##~~~~~~~~~~~~~ Hacemos one_hot_encoding y pasamos a matrices ralas ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train_val_eval <- ads_data$train_val_eval
cols_to_delete <- c("date", "title", "description", "short_description", "development_name","train_val_eval")
columns_to_keep <- setdiff(names(ads_data), cols_to_delete)

ads_data_sparse <- one_hot_sparse(ads_data[, ..columns_to_keep])
gc()

#saveRDS(ads_data_sparse,"ads_data_sparse_fea_eng_50pct.RDS")
#ads_data_sparse <- readRDS("ads_data_sparse_fea_eng_50pct.RDS")

##~~~~~~~~~~~~~ Entrenamos modelo de xgboost ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dtrain <- xgb.DMatrix(data = ads_data_sparse[train_val_eval == "train", colnames(ads_data_sparse) != "contacts"],
                      label = ads_data_sparse[train_val_eval == "train", colnames(ads_data_sparse) == "contacts"])

dvalid <- xgb.DMatrix(data = ads_data_sparse[train_val_eval == "valid", colnames(ads_data_sparse) != "contacts"],
                      label = ads_data_sparse[train_val_eval == "valid", colnames(ads_data_sparse) == "contacts"])

rgrid <- random_grid(size = 30,
                     min_nrounds = 150, max_nrounds = 350, #cantidad de arboles (0,Inf)
                     min_max_depth = 5, max_max_depth = 30, #profundidad de los arboles (0,Inf)
                     min_eta = 0.01, max_eta = 0.1, #learning rate (0,1]
                     min_gamma = 0, max_gamma = 5, #regularizador de complejidad del modelo (0,Inf)
                     min_min_child_weight = 20, max_min_child_weight = 300, #numero minimo de obs en una hoja para crear hijo (0,Inf)
                     min_colsample_bytree = 0.4, max_colsample_bytree = 0.8, #columnas sampleadas por arbol (0,1]
                     min_subsample = 0.4, max_subsample = 0.8) #observaciones sampleadas por arbol (0,1]

predicted_models <- train_xgboost(dtrain, dvalid, rgrid)

#saveRDS(predicted_models, "predicted_models_hyp_only.RDS")
#predicted_models <- readRDS("predicted_models_hyp_only.RDS")

# Guardamos los resultados en un dataframe de una forma comoda de verlo
res_table <- result_table(predicted_models)
print(res_table)

#Nos quedamos con el mejor modelo
best_model <- predicted_models[[res_table[1,"i"]]]$model

#Analizamos las variables con mayor poder predictivo
importance_matrix = xgb.importance(colnames(dtrain), model = best_model)
xgb.plot.importance(importance_matrix[1:30,])

##~~~~~~~~~~~~~ Reentrenamos con train + validation para predecir eval ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dall <- xgb.DMatrix(data = ads_data_sparse[train_val_eval != "eval", colnames(ads_data_sparse) != "contacts"],
                    label = ads_data_sparse[train_val_eval != "eval", colnames(ads_data_sparse) == "contacts"])
final_model <- xgb.train(data = dall,
                         nrounds = res_table[1, "nrounds"],
                         params=as.list(res_table[1, c("max_depth",
                                                       "eta",
                                                       "gamma",
                                                       "colsample_bytree",
                                                       "subsample",
                                                       "min_child_weight")]),
                         watchlist = list(train = dall),
                         objective = "reg:squaredlogerror",
                         feval = rmsle,
                         print_every_n = 10)

##Predecimos en eval y guardamos submissions
preds <- data.frame(ad_id = ads_data[train_val_eval == "eval", "ad_id"],
                    contacts = predict(final_model,
                                       ads_data_sparse[train_val_eval == "eval", colnames(ads_data_sparse) != "contacts"]))
preds$contacts <- pmax(preds$contacts, 0)

options(scipen=10)
write.table(preds, "submission_202107_fea_hyp_9May22_1.csv", sep=",", row.names=FALSE, quote=FALSE)
options(scipen=0)

