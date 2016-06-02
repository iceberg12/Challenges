library(readr)
library(dplyr)
library(stringr)
library(caret)
library(DMwR)
#***************************************
# functions

calWeight <- function(str){
  a <- as.numeric(unlist(strsplit(str, "[^0-9.]+")))
  a <- a[which(!is.na(a))]
  value <- a[1]
  # calculate value
  if (length(a)>=2){
    if (grepl(" x ", str)){
      value <- prod(a)
    }
    if (grepl(" + ", str)){
      value <- sum(a)
    }
  }
  #translate weight
  if (!grepl(" ml", str) & !grepl(" g", str) & !grepl("sheet", str)){
    if (grepl(" l", str) | grepl(" kg", str)){
      value <- value*1000
    }
    if (grepl("pack", str)){
      value <- 1000
    }
  }
  return (value)
}

calQuant <- function(str){
  a <- as.numeric(unlist(strsplit(str, "[^0-9.]+")))
  value <- 1
  # calculate quantity
  if (grepl(" x ", str)){
    value <- prod(a[1:length(a)-1])
  }
  return (value)
}

strToBool <- function(str){
  str <- tolower(str)
  return (str == "true")
}

#***************************************
# read data
df <- read_csv("train.csv")

#***************************************
# extra and create new features
strip_js <- function(str){
  rsl0 <- gsub(",\"", "|", tolower(str))
  rsl1 <- gsub("\"", "", rsl0)
  rsl2 <- gsub("\\{\\}", "null", rsl1)
  rsl3 <- gsub("\\{|\\}", "|", rsl2)
}
df1 <- df %>%
  mutate(js = strip_js(json_data)) 

features <- lapply(df1$js, function(str){
  gsub("\\:|\\|", "", unlist(regmatches(str, gregexpr("\\|\\w+\\:", str))))
})
m <- unique(unlist(features))

# after looking at the unavailable product data and checking statistics of these additional feature,
# not all features seems useful.
df2 <- df1 %>%
  mutate(size = gsub("size\\:|\\|", "", str_match(js, "size\\:.+?\\|")),  #good
         unit = gsub("unit\\:|\\|", "", str_match(js, "unit\\:.+?\\|")),
         brand = gsub("brand\\:|\\|", "", str_match(js, "brand\\:.+?\\|")),
         # productname = gsub(".*productname\\:|\\|.*", "", js),
         # product_unit_quantity = gsub(".*product_unit_quantity\\:|\\|.*", "", js),
         # halal = strToBool(gsub("halal\\:|\\|", "", str_match(js, "halal\\:.+?\\|"))),
         # kosher = strToBool(gsub(".*kosher\\:|\\|.*", "", js)),
         organic = strToBool(gsub("organic\\:|\\|", "", str_match(js, "organic\\:.+?\\|"))) 
         | grepl("organic", str_match(js, "productname\\:.+?\\|")),
         # vegetarian = strToBool(gsub(".*vegetarian\\:|\\|.*", "", js)),
         # description_html = gsub(".*description_html\\:|\\|.*", "", js),
         # product_info_description = gsub(".*product_info_description\\:|\\|.*", "", js),
         # description = gsub(".*description\\:|\\|.*", "", js),
         # ingredients = gsub(".*ingredients\\:|\\|.*", "", js),
         # nutritional_data = gsub(".*nutritional_data\\:|\\|.*", "", js),
         # preparation_info = gsub(".*preparation_info\\:|\\|.*", "", js),
         # special_remarks = gsub(".*special_remarks\\:|\\|.*", "", js),
         cold_storage = grepl("chill", str_match(js, "storage\\:.+?\\|")) | 
           ((grepl("frid|frig", str_match(js, "storage\\:.+?\\|"))) & !grepl("not", str_match(js, "storage\\:.+?\\|")))
         # promotion_expiry_date = gsub(".*promotion_expiry_date\\:|\\|.*", "", js),
         # published = gsub(".*published\\:|\\|.*", "", js),
         # product_info = gsub(".*product_info\\:|\\|.*", "", js),
         # directions = gsub(".*directions\\:|\\|.*", "", js),
         # short_description = gsub(".*short_description\\:|\\|.*", "", js),
         # left = gsub(".*left\\:|\\|.*", "", js),
         # right = gsub(".*right\\:|\\|.*", "", js),
         # table_description = gsub(".*table_description\\:|\\|.*", "", js),
         # name = gsub(".*name\\:|\\|.*", "", js),
         # product_info_html = gsub(".*product_info_html\\:|\\|.*", "", js),
         # volume_discount = gsub(".*volume_discount\\:|\\|.*", "", js),
  )

df2$json_data <- NULL
df2$js <- NULL

df2$unit[is.na(df2$unit) & grepl("^.+g$",df2$size),] <- 'grams'
df2$unit[is.na(df2$unit)] <- 'item'
df2$department_name <- gsub("s", "", tolower(df2$department_name))
df2$department_name <- gsub("[^[:alnum:][:space:]]", "", df2$department_name)
df2$category_name <- gsub("s", "", tolower(df2$category_name))
df2$category_name <- gsub("[^[:alnum:][:space:]]", "", df2$category_name)
df2$organic[is.na(df2$organic)] <- F

# estimate weight and product-bundle quantity from size
df3 <- df2 %>%
  rowwise() %>%
  mutate(weight = calWeight(size)) %>%
  mutate(quantity = calQuant(size))
df3$quantity[is.na(df3$quantity)] <- 1

df3 <- dplyr::select(df3, available, brand_id, store_id, department_name, category_name,
                     unit, brand, organic, cold_storage, quantity)

# convert most features to factor
df3$available <- as.factor(df3$available)
df3$brand_id <- as.factor(df3$brand_id)
df3$store_id <- as.factor(df3$store_id)
df3$department_name <- as.factor(df3$department_name)
df3$category_name <- as.factor(df3$category_name)
df3$unit <- as.factor(df3$unit)
df3$brand <- as.factor(df3$brand)
df3$organic <- as.factor(df3$organic)
df3$cold_storage <- as.factor(df3$cold_storage)

#***************************************
# prepare train, validation sets (use cross-validation and validate later)
aprod <- which(df3$available == 1)
naprod <- which(df3$available == 0)

set.seed(12); train_aprod <- sample(aprod, length(aprod)*0.8)
set.seed(12); valid_aprod <- setdiff(aprod, train_aprod)
set.seed(12); train_naprod <- sample(naprod, length(naprod)*0.8)
set.seed(12); valid_naprod <- setdiff(naprod, train_naprod)

train <- df3[c(train_aprod, train_naprod),]
valid <- df3[c(valid_aprod, valid_naprod),]
levels(train$available) <- c("yes","no")
levels(valid$available) <- c("yes","no")

train <- SMOTE(available ~ ., data = as.data.frame(train), perc.over = 300, k=5, perc.under = 170)
#***************************************
# modeling

# metric
calCost <- function(data,lev = NULL,model = NULL){
  real <- data$obs
  pred <- data$pred
  prec <- length(which(real=="yes" & pred=="yes")) / 
    (length(which(real=="yes" & pred=="yes")) + length(which(real=="no" & pred=="yes")))
  rec <- length(which(real=="yes" & pred=="yes")) / 
    (length(which(real=="yes" & pred=="yes")) + length(which(real=="yes" & pred=="no")))
  ## F_measure, emphasize recall
  # punish predicting 1 for unavailable items
  out <- 5*prec*rec/(4*prec + rec)
  names(out) <- 'customedCost'
  return(out)
}

# random forest -- tuning
library(ranger)
writeLines("Training random forest. Parameters: mtry.\n")
grid <- expand.grid(mtry=c(2, 3, 5, 9))
control <- trainControl(method = "cv", number = 9, summaryFunction = calCost, allowParallel = TRUE)

ntree = 10
ranger_train_10 <- train(available ~ ., data = train, method = "ranger", num.trees=ntree, 
                         tuneGrid = grid, trControl = control, metric = "customedCost")
ntree = 100
ranger_train_100 <- train(available ~ ., data = train, method = "ranger", num.trees=ntree, 
                         tuneGrid = grid, trControl = control, metric = "customedCost")
ntree = 200
ranger_train_200 <- train(available ~ ., data = train, method = "ranger", num.trees=ntree, 
                         tuneGrid = grid, trControl = control, metric = "customedCost")
ntree = 500
ranger_train_500 <- train(available ~ ., data = train, method = "ranger", num.trees=ntree, 
                         tuneGrid = grid, trControl = control, metric = "customedCost")
plot(ranger_train_10)
plot(ranger_train_100)
plot(ranger_train_200)
plot(ranger_train_500)
# conclude that mtry=9 is the best option, and ntree=500

# build ranger again from optimal parameter
ranger_model <- ranger(available ~ ., data=train, num.trees=500,mtry=9, 
                       importance = "impurity", 
                       write.forest = T, seed = 12)
result <- data.frame(obs=valid$available, pred=predict(ranger_model, data=valid, seed = 12)$predictions)
calCost(result)
table(pred = result$pred, real = result$obs)
writeLines("Result")
confusionMatrix(result$pred, result$obs)
writeLines("Importance variables")
data.frame(sort(importance(ranger_model)))

#**************************************
# train with 5-fold CV
control <- trainControl(method = "cv", number = 5, 
                        classProbs = T, summaryFunction = twoClassSummary, allowParallel = TRUE)
ntree = 500
ranger_train_500 <- train(available ~ ., data = train, method = "ranger", num.trees=ntree, 
                          tuneGrid = grid, trControl = control)
ranger_train_500

ranger_model <- ranger(available ~ ., data=train, num.trees=500,mtry=9, 
                       probability = T, importance = "impurity", 
                       write.forest = T, seed = 12)
test <- predict(ranger_model, data=valid, seed = 12)$predictions
colAUC(test, valid$available, plotROC = T)
# test AUC = 0.7339784
