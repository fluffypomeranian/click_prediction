########################### Readme ###############################
#
#   Author:       Tim Siwula
#   Proposal:     http://bit.ly/2gcCLQ4
#   Kaggle:       http://bit.ly/2gMVpPG
#   Github:       http://bit.ly/2gZoTwy
#   Data:         http://bit.ly/2fQ0LHW
#
##################################################################

---
output: pdf_document
---
########################### Readme ################################
#   Tim Siwula
#   https://docs.google.com/document/d/1wjOgT-j9TNjEs1zHis4oPuDpGU3SnQo6uehtbRwml3c/edit?ts=581cd593
#   https://www.kaggle.com/c/outbrain-click-prediction/data
#   https://github.com/tcsiwula/click_prediction
#
# clear workspace ----> rm(list = ls())
#
####################################################################################

library(knitr)
library(markdown)

#transform the .Rmd to a markdown (.md) file.
knit('tims_script.Rmd')

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
## 1           1  42337       0
## 2           1 139684       0
## 3           1 144739       1
## 4           1 156824       0
## 5           1 279295       0
## 6           1 296965       0
## 7           2 125211       0
## 8           2 156535       0
## 9           2 169564       0
## 10          2 308455       1
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
## 1          6 180923       0     1151028       74       0.09882334
## 2          6 180923       0     1151028       16       0.17285361
## 3         48 180923       0     1151028       74       0.09882334
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
getMergedTable="select * from merged_table limit 500000"

new_table = dbGetQuery(connection, getMergedTable)

# list the structure of mydata
str(new_table)
```

```
## 'data.frame':	500000 obs. of  6 variables:
##  $ display_id      : int  13339482 13339482 13339490 13339518 13339520 13339520 13339520 13339520 13339520 13339520 ...
##  $ ad_id           : int  180923 180923 526553 444495 389774 389774 389774 389774 389774 389774 ...
##  $ clicked         : int  0 0 1 0 1 1 1 1 1 1 ...
##  $ document_id     : int  1151028 1151028 2436524 737710 1804537 1804537 1804537 1804537 1804537 1804537 ...
##  $ topic_id        : int  74 16 197 291 159 211 16 66 184 97 ...
##  $ confidence_level: num  0.09882 0.17285 0.02792 0.22343 0.00802 ...
```
####################################################################################

##############################################################################
3) Models and response
Models to consider:
Start with the classification problem first - easier problem - start with decision trees, random forests and bagging
Then switch to regression

```r
require(tree)
```

```
## Loading required package: tree
```

```r
str(new_table)
```

```
## 'data.frame':	500000 obs. of  6 variables:
##  $ display_id      : int  13339482 13339482 13339490 13339518 13339520 13339520 13339520 13339520 13339520 13339520 ...
##  $ ad_id           : int  180923 180923 526553 444495 389774 389774 389774 389774 389774 389774 ...
##  $ clicked         : int  0 0 1 0 1 1 1 1 1 1 ...
##  $ document_id     : int  1151028 1151028 2436524 737710 1804537 1804537 1804537 1804537 1804537 1804537 ...
##  $ topic_id        : int  74 16 197 291 159 211 16 66 184 97 ...
##  $ confidence_level: num  0.09882 0.17285 0.02792 0.22343 0.00802 ...
```

```r
hist(new_table$confidence_level)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-1.png)

```r
hist(new_table$clicked)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-2.png)

```r
hist(new_table$display_id)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-3.png)

```r
hist(new_table$ad_id)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-4.png)

```r
hist(new_table$document_id)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-5.png)

```r
hist(new_table$topic_id)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-6.png)

```r
attach(new_table)
confidence20=ifelse(new_table$confidence_level>=0.2,"Yes","No")
str(confidence20)
```

```
##  chr [1:500000] "No" "No" "No" "Yes" "No" "No" "No" ...
```

```r
new_table=data.frame(new_table, confidence20)
tree.local.train.c20=tree(confidence20~.-new_table$confidence_level,data=new_table)
summary(tree.local.train.c20)
```

```
##
## Classification tree:
## tree(formula = confidence20 ~ . - new_table$confidence_level,
##     data = new_table)
## Variables actually used in tree construction:
## [1] "confidence_level"
## Number of terminal nodes:  2
## Residual mean deviance:  0 = 0 / 5e+05
## Misclassification error rate: 0 = 0 / 5e+05
```

```r
plot(tree.local.train.c20)
text(tree.local.train.c20,pretty=0)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4-7.png)
##############################################################################


########################### create a training set ###########################################

```r
# list the structure of mydata
str(new_table)
```

```
## 'data.frame':	500000 obs. of  7 variables:
##  $ display_id      : int  13339482 13339482 13339490 13339518 13339520 13339520 13339520 13339520 13339520 13339520 ...
##  $ ad_id           : int  180923 180923 526553 444495 389774 389774 389774 389774 389774 389774 ...
##  $ clicked         : int  0 0 1 0 1 1 1 1 1 1 ...
##  $ document_id     : int  1151028 1151028 2436524 737710 1804537 1804537 1804537 1804537 1804537 1804537 ...
##  $ topic_id        : int  74 16 197 291 159 211 16 66 184 97 ...
##  $ confidence_level: num  0.09882 0.17285 0.02792 0.22343 0.00802 ...
##  $ confidence20    : Factor w/ 2 levels "No","Yes": 1 1 1 2 1 1 1 1 1 2 ...
```

```r
partition_size = floor(0.80 * nrow(new_table)) ## 80% of the sample size
str(partition_size)
```

```
##  num 4e+05
```

```r
set.seed(123) ## set the seed to make your partition reproductible
partition_index <- sample(seq_len(nrow(new_table)), size = partition_size)
local_train_set <- new_table[partition_index, ]
local_test_set <- new_table[-partition_index, ]

# list the structure of mydata
str(local_test_set)
```

```
## 'data.frame':	100000 obs. of  7 variables:
##  $ display_id      : int  13339520 13339520 13339539 13339605 13339805 13339805 13339945 13340028 13340034 13340092 ...
##  $ ad_id           : int  389774 389774 400648 509131 528174 528174 90197 528174 180923 180923 ...
##  $ clicked         : int  1 1 0 0 0 0 0 1 0 0 ...
##  $ document_id     : int  1804537 1804537 1804537 2436524 2442282 2442282 483905 2442282 1151028 1151028 ...
##  $ topic_id        : int  184 97 66 197 260 124 140 260 16 16 ...
##  $ confidence_level: num  0.11432 0.21158 0.08875 0.02792 0.00818 ...
##  $ confidence20    : Factor w/ 2 levels "No","Yes": 1 2 1 1 1 1 1 1 1 1 ...
```

```r
# print first 10 rows of mydata
#head(mydata, n=10)
```
##############################################################################
