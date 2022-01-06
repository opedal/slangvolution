library(dplyr)
library(MXM)
library(glmm)
library(gee)
library(bnlearn)
library(ggpubr)
cor(x, y, method = c("pearson", "kendall", "spearman"))

#install.packages("Rfast2", type="binary")

#TODO: log transform for frequencies

#data <- read.csv("data/causal_data_input_pos.csv")
data  <- read.csv("/Users/alacrity/Documents/GitHub/kg_bias_detection/slangvolution-semantic-change/data/causal_data_input_pos4_binary.csv")
data  <- read.csv("/Users/alacrity/Documents/GitHub/kg_bias_detection/slangvolution-semantic-change/data/causal_data_MW_pos4binary.csv")
data <- read.csv("word-lists/causal_data_input_pos4_binary.csv")

head(data)

# make factor variable binary
data$type <- as.factor(data$type)
data$Noun_binary <- as.factor(data$Noun_binary)
data$Verb_binary <- as.factor(data$Verb_binary)
data$Adj_binary <- as.factor(data$Adj_binary)
data$Adverb_binary <- as.factor(data$Adverb_binary)
data$most_common <- as.factor(data$most_common)
#data$polysemy <- as.double(data$polysemy)
#data$polysemy[data$polysemy >= 10] <- 10
#data$polysemy <- factor(data$polysemy, ordered = TRUE)

data$logfreqchange <- log(data$freq2020/data$freq2010)
data$freq2010log <- log(data$freq2010)
data$freq2020log <- log(data$freq2020)
data$meanfreq <- (data$freq2010+data$freq2020)/2
data$meanfreqlog <- log(data$meanfreq)
data$Noun_binary <- as.factor(data$Noun_binary)
data$Verb_binary <- as.factor(data$Verb_binary)
data$Adj_binary <- as.factor(data$Adj_binary)
data$Adverb_binary <- as.factor(data$Adverb_binary)

THRESH = 5
data$polysemy.cat <- 0
data$polysemy.cat[data$polysemy == 1] <- "one"
data$polysemy.cat[(data$polysemy <= THRESH) & (data$polysemy > 1)] <- "few"
data$polysemy.cat[data$polysemy > THRESH] <- "many"

t1<- 3
t2 <- 5
data$polysemy.cat <- 0
data$polysemy.cat[data$polysemy == 1] <- "one"
data$polysemy.cat[(data$polysemy <= t1) & (data$polysemy > 1)] <- "few"
data$polysemy.cat[(data$polysemy <= t2) & (data$polysemy > t1)] <- "more"
data$polysemy.cat[data$polysemy > t2] <- "many"
data$polysemy.cat <- as.factor(data$polysemy.cat)

t1 <- 10
data$polysemy.cat <- 0
data$polysemy.cat[data$polysemy == 1] <- "one"
data$polysemy.cat[(data$polysemy <= t1) & (data$polysemy > 1)] <- "few"
data$polysemy.cat[data$polysemy > t1] <- "many"
data$polysemy.cat <- as.factor(data$polysemy.cat)
data$pos = as.factor(data$most_common)

## Perform gaussian test
ggdensity(data$meanfreqlog, main="Log mean frequency")
ggdensity(data$meanfreq, main="Mean frequency") # not normal
ggdensity(data$logfreqchange, main="Log frequency change")
ggdensity(data$semantic_change, main="Semantic change score")
ggdensity(data$polysemy, main="Polysemy") # not normal
ggdensity(log(data$polysemy), main="Log Polysemy")

ggqqplot(data$meanfreqlog)
ggqqplot(data$meanfreq) # not normal
ggqqplot(data$logfreqchange)
ggqqplot(data$semantic_change)
ggqqplot(data$polysemy) # not normal

data %>%
  select(freq2010, freq2020, semantic_change, pos, polysemy, polysemy.cat, type) -> data.standard

data %>%
  select(meanfreq, logfreqchange, semantic_change, pos, polysemy, polysemy.cat, type) -> data.relative

data %>%
  select(freq2010, freq2020, meanfreq, logfreqchange, semantic_change, pos,
         polysemy, polysemy.cat, type) -> data.relative2

data %>%
  select(freq2010log, freq2020log, meanfreqlog, logfreqchange, semantic_change,pos, 
         polysemy.cat, type) -> data.relativelog

data %>%
  select(meanfreqlog, logfreqchange, semantic_change,pos, 
         polysemy.cat, type) -> data.withpos

data %>%
  select(meanfreqlog, logfreqchange, semantic_change,
         Noun_binary, Verb_binary, Adj_binary, Adverb_binary,
         polysemy.cat, type) -> data.withposbin

data %>%
  select(meanfreqlog, logfreqchange, semantic_change,
         Noun_binary, 
         polysemy.cat, type) -> data.withnoun

data %>%
  select(meanfreqlog, logfreqchange, semantic_change, 
         polysemy.cat, type) -> data.nopos

data %>%
  select(freq2010log, freq2020log, meanfreqlog, logfreqchange, semantic_change, pos,
  select(meanfreqlog, logfreqchange, semantic_change, 
         polysemy.cat, type) -> data.relativelog.cat

data %>%
  select(meanfreqlog, logfreqchange, semantic_change, 
         polysemy.cat, type, Noun_binary, Verb_binary, 
         Adj_binary, Adverb_binary) -> data.pos

data %>%
  select(meanfreqlog, logfreqchange, semantic_change, 
         polysemy.cat, type, most_common) -> data.pos.mostcommon
  
data.matrix <- as.matrix(data.pos)

testIndOrdinal(factor(data.matrix[,6], ordered=TRUE), 
               dataset = data.matrix, xIndex=3, csIndex=0)

#data.scaled <- scale(data[,"freq2010", "freq2020", "semantic_change"])


#MXM::cond.regs(target=data$freq2010, dataset=as.data.frame(data[,2:4]), 
#                  xIndex=1, csIndex=0, test=testIndReg)

ci.test(x="polysemy.cat", y="semantic_change",
        data = data.relativelog.cat, test="mi-cg")
ci.test(x="meanfreqlog", y="semantic_change", z = c("polysemy.cat","type"),
        data = data.relativelog.cat, test="mi-cg")

# correlation between continuous polysemy and log freq
ci.test(x="polysemy", y="meanfreqlog",
        data = data.relativelog, test="cor")
ci.test(x="polysemy", y="meanfreq",
        data = data.relative2, test="cor")


ci.test(x="polysemy", y="meanfreqlog", data = data.relativelog, test="mc-mi-g")
#mi, mi-adf, mc-mi, smc-mi, sp-mi, mi-sh, x2, x2-adf, mc-x2, smc-x2, sp-x2
#jt, mc-jt, smc-jt
#cor, mc-cor, smc-cor, zf, mc-zf, smc-zf, mi-g, mc-mi-g, smc-mi-g, mi-g-sh
#mi-cg

res <- pc.stable(data.withpos, alpha=0.05)
plot(res)

res <- pc.stable(data.withposbin, alpha=0.05)
plot(res)

res <- pc.stable(data.withnoun, alpha=0.05)
plot(res)

res <- pc.stable(data.nopos, alpha=0.05)
plot(res)


res <- gs(data.withposbin)
plot(res)

res <- gs(data.withpos)
plot(res)

res <- gs(data.withnoun)
plot(res)

res <- gs(data.nopos)
plot(res)

res <- fast.iamb(data.withposbin)
plot(res)

res <- mmpc(data.withpos)
plot(res)

res <- si.hiton.pc(data.withposbin)
plot(res)

res <- si.hiton.pc(data.nopos)
plot(res)

res <- hpc(data.withposbin)
res <- pc.stable(data.pos, alpha = 0.05)
plot(res)

res <- pc.stable(data.pos, alpha = 0.03)
plot(res)

res <- pc.stable(data.pos, alpha = 0.01)
plot(res)

res <- pc.stable(data.pos, test="jt,mi")
plot(res)

res <- gs(data.relativelog.cat)
plot(res)

res <- fast.iamb(data.relativelog.cat)
plot(res)

res <- mmpc(data.relativelog.cat)
plot(res)

res <- si.hiton.pc(data.relativelog.cat)
plot(res)

res <- hpc(data.relativelog.cat)
plot(res)

res <- hc(data.relative2)
plot(res)

res <- tabu(data.relative2)
plot(res)
