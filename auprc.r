# auprc.r
# usage: Rscript auprc.r
# calculate the Area under Precision Recall curve

library(zoo)

# order x in increasing fashion
x <- c(0.755239138372, 0.761591821833, 0.766995253742, 0.772033588901, 0.778386272362, 0.783935742972, 0.789558232932, 0.795618838992, 0.801533406353)
y <- c(1, 1, 1, 1, 1, 0.999255398362, 0.995305596465, 0.987493202828, 0.9801768015)
id <- order(x)

auc <- sum(diff(x[id]) * rollmean(y[id], 2))
print(paste0("auPRC = ", auc))
