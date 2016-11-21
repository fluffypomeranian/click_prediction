---
output: pdf_document
---
#   Tim Siwula
   https://docs.google.com/document/d/1wjOgT-j9TNjEs1zHis4oPuDpGU3SnQo6uehtbRwml3c/edit?ts=581cd593
   https://www.kaggle.com/c/outbrain-click-prediction/data
   https://github.com/tcsiwula/click_prediction
#
 clear workspace ----> rm(list = ls())
#
########################### SET UP DATABASE CONNECTION ################################

```r
require("RPostgreSQL")    #install.packages("RPostgreSQL")
driver <- dbDriver("PostgreSQL")   # loads the PostgreSQL driver

# creates a connection to the postgres database
# note that "con" will be used later in each connection to the database
connection <- dbConnect(driver, dbname = "clickprediction",
                 host = "localhost", port = 5432,
                 user = "admin", password = "admin")
dbExistsTable(connection, "clicks_train")  # confirm the tables are accessible
```

```
## [1] TRUE
```
####################################################################################

########################### QUERY THE DATABASE ########################################

```r
# 1) 
# try to find features related to ad_id.
# here we join click_train and promoted-content with ad_id.

# look at clicks_train first
getClicksTrain="select * from clicks_train limit 10 "
clicks_train = dbGetQuery(connection, getClicksTrain)
clicks_train
```

```
##    display_id  ad_id clicked
## 1    14167986 190281       1
## 2    14167986 281177       0
## 3    14167986 285834       0
## 4    14167986 473807       0
## 5    14167987  39896       1
## 6    14167987 174547       0
## 7    14167987 196715       0
## 8    14167987 213436       0
## 9    14167987 213509       0
## 10   14167987 232273       0
```

```r
# look at promoted_content next
getPromotedContent="select * from promoted_content limit 10"
promoted_content = dbGetQuery(connection, getPromotedContent)
promoted_content
```

```
##    ad_id document_id campaign_id advertiser_id
## 1      1        6614           1             7
## 2      2      471467           2             7
## 3      3        7692           3             7
## 4      4      471471           2             7
## 5      5      471472           2             7
## 6      6       12736           1             7
## 7      7       12808           1             7
## 8      8      471477           2             7
## 9      9       13379           1             7
## 10    10       13885           1             7
```

```r
# join click_train and promoted-content with ad_id new table
# 500k apears to be stable with rstudio.
join_query = "
select t.display_id, t.ad_id, t.clicked, d.document_id,
d.topic_id, d.confidence_level
from clicks_train t, promoted_content p, documents_topics d
where t.ad_id = p.ad_id and p.document_id = d.document_id 
limit 500000;"
merged_table=dbGetQuery(connection, join_query)
head(merged_table, 3)
```

```
##   display_id  ad_id clicked document_id topic_id confidence_level
## 1   14167988 180923       0     1151028       74       0.09882334
## 2   14167988 180923       0     1151028       16       0.17285361
## 3   14168039 526553       0     2436524      197       0.02791654
```

```r
dim(merged_table)
```

```
## [1] 500000      6
```
####################################################################################

########################### Create new table -- write new table ########################################

```r
dbWriteTable(connection, "merged_table", merged_table, row.names=FALSE)
```

```
## Warning in postgresqlWriteTable(conn, name, value, ...): table merged_table
## exists in database: aborting assignTable
```

```
## [1] FALSE
```

```r
# look at the new table
getMergedTable="select * from merged_table limit 10"
new_table = dbGetQuery(connection, getMergedTable)
new_table
```

```
##    display_id  ad_id clicked document_id topic_id confidence_level
## 1    13339482 180923       0     1151028       74      0.098823340
## 2    13339482 180923       0     1151028       16      0.172853612
## 3    13339490 526553       1     2436524      197      0.027916545
## 4    13339518 444495       0      737710      291      0.223427652
## 5    13339520 389774       1     1804537      159      0.008018133
## 6    13339520 389774       1     1804537      211      0.009418253
## 7    13339520 389774       1     1804537       16      0.062539061
## 8    13339520 389774       1     1804537       66      0.088745137
## 9    13339520 389774       1     1804537      184      0.114324378
## 10   13339520 389774       1     1804537       97      0.211581768
```
####################################################################################

########################### create a training set ###########################################

```r
partition_size = floor(0.80 * nrow(clicks_train)) ## 80% of the sample size
set.seed(123) ## set the seed to make your partition reproductible
partition_index <- sample(seq_len(nrow(clicks_train)), size = partition_size)
local_train_set <- clicks_train[partition_index, ]
local_test_set <- clicks_train[-partition_index, ]
dim(local_train_set)
dim(local_test_set)
'''
##############################################################################

########################### probability calculation ###########################################
library(data.table)

# take care , cant be variables with the same name as var or target in dt... 
#if you have a beter implamentation of this functions,share it pls ^^
get_probs <- function (dt,var,target,w){
  p=dt[,sum(get(target))/.N]
  dt[ ,.( prob=(sum(get(target))+w*p )/(.N+w) ),by=eval(var)]
}
DT_fill_NA <- function(DT,replacement=0) {
  for (j in seq_len(ncol(DT)))
    set(DT,which(is.na(DT[[j]])),j,replacement)
}
super_fread <- function( file , key_var=NULL){
  dt <- fread(file)
  if(!is.null(key_var)) setkeyv(dt,c(key_var))
  return(dt)
}

#
clicks_train  <- super_fread( "data/clicks_train.csv", key_var = "ad_id" )

#
click_prob = clicks_train[,.(sum(clicked)/.N)]
ad_id_probs   <- get_probs(clicks_train,"ad_id","clicked",8)
#dim(ad_id_probs)
rm(clicks_train)
gc()

clicks_test   <- super_fread( "data/clicks_test.csv" , key_var = "ad_id" )
clicks_test <- merge( clicks_test, ad_id_probs, all.x = T )

DT_fill_NA( clicks_test, click_prob )

setkey(clicks_test,"prob")
submission <- clicks_test[,.(ad_id=paste(rev(ad_id),collapse=" ")),by=display_id]
setkey(submission,"display_id")

write.csv(submission,file = "submission.csv",row.names = F)
############################################################################
```

```
## Error: <text>:8:3: unexpected INCOMPLETE_STRING
## 49: write.csv(submission,file = "submission.csv",row.names = F)
## 50: ############################################################################
##       ^
```
