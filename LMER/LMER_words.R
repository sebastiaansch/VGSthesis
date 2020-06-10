library('lme4')
library('lmerTest')
library('dfoptim')
library('ordinal')
data = read.csv("/Users/sebastiaanscholten/Downloads/rfilecsv.csv")
# data1 = read.csv("/Users/sebastiaanscholten/Documents/speech2image-master/vgsexperiments/experiments/Results_isolated_word_recognition/wordinstancesdata.csv")
# data1$WordId = data$WordId
# data1$spk_id = data$spk_id
# data1$P10=data1$Precision.10
# data1$TrainingSetOccurence = scale(data$TrainingSetOccurence,center=TRUE,scale=TRUE)
# data1$MfccLength = scale(data$MfccLength,center=TRUE,scale=TRUE)
# data1$SpeakingSpeed = scale(data$SpeakingSpeed,center=TRUE,scale=TRUE)
# data1$NofConsonants = scale(data$NofConsonants,center=TRUE,scale=TRUE)
# data1$NofVowels = scale(data$NofVowels,center=TRUE,scale=TRUE)
# data1$NofPhonemes = scale(data$NofPhonemes,center=TRUE,scale=TRUE)


f1 <- P10 ~ MfccLength + SpeakingSpeed +
  TrainingSetOccurence *(NofVowels + NofPhonemes + NofConsonants) +
  (1 | spk_id) +
  (1 | WordId) +
  (0 + MfccLength | spk_id) 
# data1$P10 <- as.numeric(factor(data1$P10, ordered = TRUE))
# data$P10 <- factor(data$P10, ordered = TRUE)

m1 <- lmer(f1, data, REML = F, control = lmerControl(optimizer = 'bobyqa'))

summary(m1)

ss <- getME(m1,c('theta','fixef'))

m1.all <- allFit(f1)


