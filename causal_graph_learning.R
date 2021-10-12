library(dplyr)
library(MXM)
library(glmm)
library(gee)
library(bnlearn)

#install.packages("Rfast2", type="binary")

data <- read.csv("word-lists/causal_data_input.csv")
head(data)

data %>%
  select(freq2010, freq2020, semantic_change, polysemy, type) -> data

# make factor variable binary
data$type <- as.factor(data$type)
#data$polysemy <- as.double(data$polysemy)

head(data)

MXM::cond.regs(target=data$freq2010, dataset=as.data.frame(data[,2:4]), 
                  xIndex=1, csIndex=0, test=testIndReg)
browser()

ci.test("freq2010", "freq2020", data = data, test = "cor")

res <- pc.stable(data, alpha=0.05)
plot(res)

res <- pc.stable(data≈)
plot(res)

res <- gs(data)
plot(res)

res <- fast.iamb(data)
plot(res)

res <- mmpc(data)
plot(res)

res <- si.hiton.pc(data)
plot(res)

res <- hpc(data)
plot(res)

