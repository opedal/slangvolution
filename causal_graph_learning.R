library(dplyr)
library(MXM)
library(glmm)
library(gee)
library(bnlearn)
library(ggpubr)

froot = "data/"

## Import
data  <- read.csv(paste(froot,"causal_data_input.csv", sep=""))
head(data)

## Preprocess
data$type <- as.factor(data$type)
data$Noun_binary <- as.factor(data$Noun_binary)
data$Verb_binary <- as.factor(data$Verb_binary)
data$Adj_binary <- as.factor(data$Adj_binary)
data$Adverb_binary <- as.factor(data$Adverb_binary)
data$most_common <- as.factor(data$most_common)

data$logfreqchange <- log(data$freq2020/data$freq2010)
data$freq2010log <- log(data$freq2010)
data$freq2020log <- log(data$freq2020)
data$meanfreq <- (data$freq2010+data$freq2020)/2
data$meanfreqlog <- log(data$meanfreq)

## Plot
ggdensity(data$meanfreqlog, xlab="Log frequency")
ggdensity(data$meanfreq, xlab="Mean frequency") # not normal
ggdensity(data$logfreqchange, xlab="Log frequency change")
ggdensity(data$semantic_change, xlab ="Semantic change score")
ggdensity(data$normalized_semantic_change, xlab="Normalized semantic change score")
ggdensity(data$polysemy, xlab="Polysemy") # not normal
ggdensity(log(data$polysemy), xlab="Log Polysemy")

ggqqplot(data$meanfreqlog, xlab="Log frequency")
ggqqplot(data$meanfreq) # not normal
ggqqplot(data$logfreqchange, xlab="Log frequency change")
ggqqplot(data$normalized_semantic_change, xlab="Normalized semantic change score")
ggqqplot(data$polysemy) # not normal

## Causal discovery

data %>%
  select(meanfreqlog, logfreqchange, semantic_change, 
         polysemy.cat, type, Noun_binary, Verb_binary, 
         Adj_binary, Adverb_binary) -> data.causal
  
data.matrix <- as.matrix(data.causal)

testIndOrdinal(factor(data.matrix[,6], ordered=TRUE), 
               dataset = data.matrix, xIndex=3, csIndex=0)

#ci.test(x="polysemy.cat", y="semantic_change",
#        data = data.relativelog.cat, test="mi-cg")
#ci.test(x="meanfreqlog", y="semantic_change", z = c("polysemy.cat","type"),
#        data = data.relativelog.cat, test="mi-cg")
#
# correlation between continuous polysemy and log freq
#ci.test(x="polysemy", y="meanfreqlog",
#        data = data.relativelog, test="cor")
#ci.test(x="polysemy", y="meanfreq",
#        data = data.relative2, test="cor")


#ci.test(x="polysemy", y="meanfreqlog", data = data.relativelog, test="mc-mi-g")
#mi, mi-adf, mc-mi, smc-mi, sp-mi, mi-sh, x2, x2-adf, mc-x2, smc-x2, sp-x2
#jt, mc-jt, smc-jt
#cor, mc-cor, smc-cor, zf, mc-zf, smc-zf, mi-g, mc-mi-g, smc-mi-g, mi-g-sh
#mi-cg

t1<- 2
t2 <- 5
data$polysemy.cat <- 0
data$polysemy.cat[data$polysemy == 1] <- "one"
data$polysemy.cat[(data$polysemy <= t1) & (data$polysemy > 1)] <- "few"
data$polysemy.cat[(data$polysemy <= t2) & (data$polysemy > t1)] <- "more"
data$polysemy.cat[data$polysemy > t2] <- "many"
data$polysemy.cat <- factor(data$polysemy.cat,ordered = TRUE, 
                            levels = c("one", "few", "more", "many"))

t1 <- 2
data$polysemy.cat <- 0
data$polysemy.cat[data$polysemy == 1] <- "one"
data$polysemy.cat[(data$polysemy <= t1) & (data$polysemy > 1)] <- "few"
data$polysemy.cat[data$polysemy > t1] <- "many"
data$polysemy.cat <- factor(data$polysemy.cat,ordered = TRUE,
                            levels = c("one", "few", "more", "many"))

res <- pc.stable(data.causal, alpha=0.05)
plot(res)

res <- pc.stable(data.causal, alpha=0.03)
plot(res)

res <- pc.stable(data.causal, alpha=0.01)
plot(res)
