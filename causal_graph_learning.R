library(dplyr)
library(MXM)
library(glmm)
library(gee)
library(bnlearn)
library(ggpubr)

#install.packages("Rfast2", type="binary")

#TODO: log transform for frequencies

data <- read.csv("word-lists/causal_data_input.csv")
head(data)

# make factor variable binary
data$type <- as.factor(data$type)
#data$polysemy <- as.double(data$polysemy)
data$polysemy[data$polysemy >= 10] <- 10
#data$polysemy <- factor(data$polysemy, ordered = TRUE)

data$logfreqchange <- log(data$freq2020/data$freq2010)
data$freq2010log <- log(data$freq2010)
data$freq2020log <- log(data$freq2020)
data$meanfreq <- (data$freq2010+data$freq2020)/2
data$meanfreqlog <- log(data$meanfreq)

data$polysemy.cat <- 0
data$polysemy.cat[data$polysemy == 1] <- "one"
data$polysemy.cat[(data$polysemy <= 3) & (data$polysemy > 1)] <- "few"
data$polysemy.cat[data$polysemy > 3] <- "many"
data$polysemy.cat <- as.factor(data$polysemy.cat)

## Perform gaussian test
ggdensity(data$meanfreqlog, main="Log mean frequency")
ggdensity(data$meanfreq, main="Mean frequency") # not normal
ggdensity(data$logfreqchange, main="Log frequency change")
ggdensity(data$semantic_change, main="Semantic change score")
ggdensity(data$polysemy, main="Polysemy") # not normal

ggqqplot(data$meanfreqlog)
ggqqplot(data$meanfreq) # not normal
ggqqplot(data$logfreqchange)
ggqqplot(data$semantic_change)
ggqqplot(data$polysemy) # not normal

data %>%
  select(freq2010, freq2020, semantic_change, polysemy, type) -> data.standard

data %>%
  select(meanfreq, logfreqchange, semantic_change, polysemy, type) -> data.relative

data %>%
  select(freq2010, freq2020, meanfreq, logfreqchange, semantic_change, 
         polysemy, type) -> data.relative2

data %>%
  select(freq2010log, freq2020log, meanfreqlog, logfreqchange, semantic_change, 
         polysemy, type) -> data.relativelog

data %>%
  select(freq2010log, freq2020log, meanfreqlog, logfreqchange, semantic_change, 
         polysemy.cat, type) -> data.relativelog.cat
  
data.matrix <- as.matrix(data.relativelog)

testIndOrdinal(factor(data.matrix[,6], ordered=TRUE), 
               dataset = data.matrix, xIndex=3, csIndex=0)

#data.scaled <- scale(data[,"freq2010", "freq2020", "semantic_change"])


#MXM::cond.regs(target=data$freq2010, dataset=as.data.frame(data[,2:4]), 
#                  xIndex=1, csIndex=0, test=testIndReg)

ci.test(x="polysemy", y="meanfreqlog", data = data.relativelog, test = "jt")
#mi, mi-adf, mc-mi, smc-mi, sp-mi, mi-sh, x2, x2-adf, mc-x2, smc-x2, sp-x2
#jt, mc-jt, smc-jt
#cor, mc-cor, smc-cor, zf, mc-zf, smc-zf, mi-g, mc-mi-g, smc-mi-g, mi-g-sh
#mi-cg

res <- pc.stable(data.relativelog.cat, alpha = 0.5)
plot(res)

res <- pc.stable(data.relativelog, test="jt,mi")
plot(res)

res <- gs(data.relativelog)
plot(res)

res <- fast.iamb(data.relativelog)
plot(res)

res <- mmpc(data.relativelog)
plot(res)

res <- si.hiton.pc(data.relativelog)
plot(res)

res <- hpc(data.relative)
plot(res)

res <- hc(data.relative2)
plot(res)

res <- tabu(data.relative2)
plot(res)