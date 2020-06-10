library('lme4')
library('lmerTest')
library('dfoptim')
library('Matrix')
library('ordinal')

data= read.csv("/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/Results_isolated_word_recognition/phonemedata.csv")
data$wordcohort = scale(data$wordcohort,center=TRUE,scale=TRUE)
data$neighbourhooddensity = scale(data$neighbourhooddensity,center=TRUE,scale=TRUE)
data$TrainingSetOccurence = scale(data$TrainingSetOccurence,center=TRUE,scale=TRUE)
data$AmntPhonemes = scale(data$AmntPhonemes,center=TRUE,scale=TRUE)

f1 <- Precision.10 ~ AmntPhonemes + wordcohort + neighbourhooddensity + 
  TrainingSetOccurence +
  (1 | spk_id) +
  (1 | wordid)

m1 <- lmer(f1, data, REML = F, control = lmerControl(optimizer = 'bobyqa'))

summary(m1)

ss <- getME(m1,c('theta','fixef'))

m1.all <- allFit(f1)
