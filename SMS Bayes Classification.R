# Load Libraries
pacman::p_load(Amelia, C50, class, corrplot, FactoMineR, MASS, RMySQL, backports, 
               caret, cellranger, corrplot, doParallel, dplyr, e1071, factoextra,
               foreach, forecast, GGally, ggfortify, ggplot2, gmodels, inum, kknn,
               padr, party, plotly, plyr, psych, randomForest, readr, reshape,
               reshape2, rio, rmeta, rstudioapi, scatterplot3d, SnowballC, 
               stats, stringr, tidyverse, tidyverse, tm, utiml, wordcloud)

# Set working directory
current_path <- getActiveDocumentContext()$path
setwd(dirname(current_path))
remove(current_path)

# Import dataset
sms_raw <- read.csv("Datasets/sms_spam.csv", stringsAsFactors = F)

# Initial Dataset Exploration
str(sms_raw)

# Convert categorical variable into factor
sms_raw$type <- as.factor(sms_raw$type)

#Dataset Exploration
str(sms_raw)
table(sms_raw$type)

# Creating a corpus
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# For potential additional options offered by the "tm" package:
#vignette("tm")

inspect(sms_corpus[1:5])

# to check for a single message we can use the as.character() function as well as double-brackets
as.character(sms_corpus[[500]])

# Using lapply() to print several messages
lapply(sms_corpus[1:10], as.character)

# converting the corpus to lower-case to start cleaning up the corpus
sms_corpus_clean <- tm_map(sms_corpus,
                           content_transformer(tolower))

# Comparing first sms to check result
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

# Removing numbers to reduce noise (numbers will be unique and will not provide useful patters across all messages)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

# Check results
lapply(sms_corpus_clean[1:10], as.character)

# Removing "stop words" (to, and, but, or) and punctuation
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

# Stemming
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

# Remove additional whitespace
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)

# Tokenization
sms_dtm <- DocumentTermMatrix(sms_corpus_clean) # Thanks to the previous preprocessing the object is ready

# Alternatively, we could have tweaked the DTM parameters such as below:
# sms_dtm2 <- DocumentTermMatrix(sms_corpus, control = list(
#   tolower = TRUE,
#   removeNumbers = TRUE,
#   stopwords = TRUE,
#   removePunctuation = TRUE,
#   stemming = TRUE
# ))

# Data preparation - Train and Test ####
sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]

sms_train_labels <- sms_raw[1:4169,]$type
sms_test_labels <- sms_raw[4170:5559,]$type

# Comparing proportion of SPAM
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

# Creating a word cloud to visualize the text data
wordcloud(sms_corpus_clean, min.freq = 50, random.order = F)

# Creating subset of SPAM sms to visualize later
spam <- subset(sms_raw, type == "spam")
ham <- subset(sms_raw, type == "ham")

# Visualizing both types separately
wordcloud(spam$text, max.words = 40, scale = c(3, 0.5), random.order = F)
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5), random.order = F)

# Data preparation - creating indicator features for frequent words####
# Filtering out unfrequent words
sms_freq_words <- findFreqTerms(sms_dtm_train, 5) # function to find all terms appearing at least 5 times
sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[,sms_freq_words]

# Changing cells in sparse matrix to indicate yes/no since Naive Bayes typically works with Categorical features
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,
                   convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,
                  convert_counts)

# Training model on the data ####
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

# Evaluating model performance
sms_test_pred <- predict(sms_classifier, sms_test)

CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = F, prop.t = F,
           dnn = c('predicted', 'actual'))

# Improving Model Performance

# Rebuilding Naive Bayes with laplace = 1
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels,
                              laplace = 1)

# Evaluating 2nd model's performance
sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = F, prop.t = F, prop.r = F,
           dnn = c('predicted', 'actual'))

