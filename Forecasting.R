

library(magrittr)
library(ggplot2)
library(tseries)
library(forecast)
library(Hmisc)
library(data.table)
#library(tidyverse)
library(dplyr)
library(pastecs)
library(sandwich)
library(lmtest)
library(ggplot2)
library(ggthemes)
library(xtable)
library(e1071)   
library(car)
library(cragg)

#h governs the forecast horizon
h = 1

#reading vix data
df_vix <- read.csv(file="source/volatility/VIXCLS.CSV")
df_vix$DATE <- as.Date(df_vix$DATE, format="%m/%d/%Y")

#reading speech data
df_speech <- read.csv(file="data_files_generated/genered_speech_sentiment.csv")
df_speech["DATE"] <- df_speech["date_manual"]
df_speech$DATE <- as.Date(df_speech$DATE, format="%Y-%m-%d")
df_speech <- df_speech %>% distinct(DATE, .keep_all = TRUE)
df_speech$is_speech <- 1
df <- merge(df_vix, df_speech, by=c("DATE"), all.x=TRUE)
df <- df[df[["DATE"]] >= "1996-09-01", ]
df <- df[df[["DATE"]] <= "2018-06-01", ]
df <- df[!is.na(df$VIXCLS), ]
df$is_speech[is.na(df$is_speech)] <- 0

#classifying each speech by the speaker type \in {chair, board of govornors, president}
chair_list <- c("Jerome H Powell", "Janet L Yellen", "Ben S Bernanke", "Alan Greenspan", "Alice M Rivlin", "Edward M Gramlich")
bog_list <- c("Daniel K Tarullo", 
              "Edward W Kelley Jr", "Elizabeth A Duke", "Frederic S Mishkin",
              "Jeremy C Stein", "Kevin M Warsh", "Lael Brainard",
              "Laurence H Meyer", "Mark W Olson", "Randall S Kroszner",
              "Robert W Ferguson Jr", "Sarah Bloom Raskin", "Susan Schmidt Bies")
pres_list <- c("Thomas M Hoenig", "Timothy F Geithner", "William C Dudley", "William J McDonough", "Brian P Sack",
               "Charles I Plosser", "Ernest T Patrikis", "James McAndrews", "Joseph S Tracy", "Narayana Kocherlakota", 
               "Simon M Potter")
for (i in 1:dim(df)){
  if (df[i,"author"] %in% chair_list){
    df[i,"speaker_chair"] <- 1
  } else if (df[i,"author"] %in% bog_list){
    df[i,"speaker_bog"] <- 1
  } else if (df[i,"author"] %in% pres_list){
    df[i,"speaker_pres"] <- 1
  }
}
df$speaker_chair[is.na(df$speaker_chair)] <- 0
df$speaker_bog[is.na(df$speaker_bog)] <- 0
df$speaker_pres[is.na(df$speaker_pres)] <- 0

#logging vix
df$VIX_nolog <- df$VIXCLS
df$VIXCLS <- log(df$VIXCLS)



#creating lagged regressors for past averages of VIX
l <- c(1, 5, 22)
for (i in 1:dim(df)){
  for (k in l){
    if (i > k){
      vaktemp <- 0
      for (j in 1:k){
        vaktemp = vaktemp + df[i-j, "VIXCLS"]
      }
      vaktemp = vaktemp / k
      df[i-1, paste("y", k, sep="")] <- vaktemp
    }
  }
}
df["y0"] <- Lag(df$VIXCLS, -h)



#ECONOMIC VARS
df_sp500 <- read.csv(file="source/volatility/SPY.csv")
df_sp500$DATE <- as.Date(df_sp500$Date, format="%m/%d/%Y")
#df_libor<- read.csv(file="source/volatility/LIBOR.csv")
#df_libor$DATE <- as.Date(df_libor$DATE, format="%m/%d/%Y")
df_10y <- read.csv(file="source/volatility/10YEAR.csv")
df_10y$DATE <- as.Date(df_10y$DATE, format="%m/%d/%Y")
df_3m <- read.csv(file="source/volatility/3MONTH.csv")
df_3m$DATE <- as.Date(df_3m$DATE, format="%m/%d/%Y")
df_oil <- read.csv(file="source/volatility/OIL.csv")
df_oil$DATE <- as.Date(df_oil$DATE, format="%m/%d/%Y")
df_credit <- read.csv(file="source/volatility/CREDIT.csv")
df_credit$DATE <- as.Date(df_credit$DATE, format="%m/%d/%Y")


df <- merge(df, df_sp500, by=c("DATE"), all.x=TRUE)
df <- merge(df, df_3m, by=c("DATE"), all.x=TRUE)
df <- merge(df, df_10y, by=c("DATE"), all.x=TRUE)
df <- merge(df, df_oil, by=c("DATE"), all.x=TRUE)
df <- merge(df, df_credit, by=c("DATE"), all.x=TRUE)

#imputing missing vars with previous day's value
var_list <- c("adjclose", "volume", "credit", "tres10y", "tres3m", "oil")
for (i in 1:dim(df)){
  for (var in var_list){
    if(is.na(df[i, var])){
      df[i, var] <- df[i-1, var]
    }
  }
}


#computing past sp500 returns for each k in l
for(k in l){
  df[,paste("d_sp_", k, sep="")] <- 100 * (df[,"adjclose"] - Lag(df[,"adjclose"], k)) / Lag(df[,"adjclose"], k)
  df[,paste("d_oil_", k, sep="")] <- 100 * (df[,"oil"] - Lag(df[,"oil"], k)) / Lag(df[,"oil"], k)
}
df[,"term_spread"] <- df[,"tres10y"] - df["tres3m"]
df[,"d_term_spread"] <- c(NA, diff(df[,"term_spread"]))
df[,"d_credit_spread"] <- c(NA, diff(df[, "credit"]))
df[,"d_volume"] <- c(NA, diff(log(df[,"volume"])))




#dropping vars lost due to averaging
df <- df[!is.na(df$d_sp_22), ]
df <- df[!is.na(df$y0), ]

var_list <- c("d_sp_1", "d_sp_5", "d_sp_22", "d_oil_1", "d_oil_5", "d_oil_22", "d_term_spread", "d_credit_spread", "d_volume" )
#var_list <- c("y0", "VIX_nolog", "volume", "term_spread", "credit", "oil")
#adf test
for(var in var_list){
  print(var)
  print(adf.test(na.omit(df[,var])))
}






df["bert_score_prob"] <- ifelse(df$bert_sent==2, df$bert_score, 1-df$bert_score)
#df["bert_fin_score_prob"] <- ifelse(df$finbert_sent==2, df$finbert_score, ifelse(df$finbert_sent == 1, 0.5, 1-df$finbert_score))
#df["roberta_score_prob"] <- ifelse(df$roberta_sent==2, df$roberta_score, ifelse(df$roberta_sent == 1, 0.5, 1-df$roberta_score))
df["roberta_score_prob"] <- ifelse(df$roberta_sent2==2, df$roberta_score2, 1-df$roberta_score2)
#df["roberta_imdb_score_prob"] <- ifelse(df$roberta_imdb_sent==2, df$roberta_imdb_score, 1-df$roberta_imdb_score)
#df["roberta_imdb2_score_prob"] <- ifelse(df$roberta_imdb_sent2==2, df$roberta_imdb_score2, 1-df$roberta_imdb_score2)
#df["xlnet_score_prob"] <- ifelse(df$xlnet_sent==2, df$xlnet_score, 1-df$xlnet_score)
df["xlnet_score_prob"] <- ifelse(df$xlnet_sent2==2, df$xlnet_score2, 1-df$xlnet_score2)
df["electra_score_prob"] <- ifelse(df$electra_sent==2, df$electra_score, 1-df$electra_score)

#creating fitted sentiment regressor based on OLS fitted values of other sentiment predictions
sent.fit.bert <- lm(data=df, bert_score_prob ~ roberta_score_prob + xlnet_score_prob + electra_score_prob + d_sp_1 + d_sp_5 + d_sp_22 + d_oil_1 + d_oil_5 + d_oil_22 + d_term_spread + d_credit_spread + d_volume)
sent.fit.roberta <- lm(data=df, roberta_score_prob ~ bert_score_prob  + xlnet_score_prob + electra_score_prob + d_sp_1 + d_sp_5  + d_sp_22 + d_oil_1+ d_oil_5  + d_oil_22 + d_term_spread + d_credit_spread + d_volume)
sent.fit.xlnet <- lm(data=df, xlnet_score_prob ~ roberta_score_prob + bert_score_prob  + electra_score_prob + d_sp_1  + d_sp_5 + d_sp_22 + d_oil_1 + d_oil_5 + d_oil_22 + d_term_spread + d_credit_spread + d_volume)
sent.fit.electra <- lm(data=df, electra_score_prob ~ roberta_score_prob + xlnet_score_prob + bert_score_prob + d_sp_1 + d_sp_5 + d_sp_22  + d_oil_1 + d_oil_5 + d_oil_22  + d_term_spread + d_credit_spread + d_volume) 
summary(sent.fit.bert)
summary(sent.fit.roberta)
summary(sent.fit.xlnet)
summary(sent.fit.electra)

print(stock_yogo_test(~d_sp_1 + d_sp_5 + d_sp_22  + d_oil_1 + d_oil_5 + d_oil_22  + d_term_spread + d_credit_spread + d_volume,
                      ~bert_score_prob,
                      ~roberta_score_prob + xlnet_score_prob + electra_score_prob ,
                      B=.1,
                      size_bias="size",
                      data=df[!is.na(df$bert_score_prob), ]))
print(stock_yogo_test(~d_sp_1 + d_sp_5 + d_sp_22  + d_oil_1 + d_oil_5 + d_oil_22  + d_term_spread + d_credit_spread + d_volume,
                      ~roberta_score_prob,
                      ~ bert_score_prob + xlnet_score_prob + electra_score_prob ,
                      B=.1,
                      size_bias="size",
                      data=df[!is.na(df$bert_score_prob), ]))
print(stock_yogo_test(~d_sp_1 + d_sp_5 + d_sp_22  + d_oil_1 + d_oil_5 + d_oil_22  + d_term_spread + d_credit_spread + d_volume,
                      ~ xlnet_score_prob,
                      ~roberta_score_prob + bert_score_prob + electra_score_prob ,
                      B=.1,
                      size_bias="size",
                      data=df[!is.na(df$bert_score_prob), ]))
print(stock_yogo_test(~d_sp_1 + d_sp_5 + d_sp_22  + d_oil_1 + d_oil_5 + d_oil_22  + d_term_spread + d_credit_spread + d_volume,
                      ~ electra_score_prob,
                      ~roberta_score_prob + xlnet_score_prob + bert_score_prob ,
                      B=.1,
                      size_bias="size",
                      data=df[!is.na(df$bert_score_prob), ]))




print(linearHypothesis(sent.fit.bert, c("roberta_score_prob=0", "xlnet_score_prob=0", "electra_score_prob=0"), white.adjust="hc3"))
print(linearHypothesis(sent.fit.roberta, c("bert_score_prob=0", "xlnet_score_prob=0", "electra_score_prob=0"), white.adjust="hc3"))
print(linearHypothesis(sent.fit.xlnet, c("roberta_score_prob=0", "bert_score_prob=0", "electra_score_prob=0"), white.adjust="hc3"))
print(linearHypothesis(sent.fit.electra, c("roberta_score_prob=0", "xlnet_score_prob=0", "bert_score_prob=0"), white.adjust="hc3"))

print(linearHypothesis(sent.fit.bert, c("d_sp_1=0", "d_sp_5=0", "d_sp_22=0", "d_oil_1=0", "d_oil_5=0", "d_oil_22=0", "d_term_spread=0", "d_credit_spread=0", "d_volume=0"), white.adjust="hc3"))
print(linearHypothesis(sent.fit.roberta, c("d_sp_1=0", "d_sp_5=0", "d_sp_22=0", "d_oil_1=0", "d_oil_5=0", "d_oil_22=0", "d_term_spread=0", "d_credit_spread=0", "d_volume=0"), white.adjust="hc3"))
print(linearHypothesis(sent.fit.xlnet, c("d_sp_1=0", "d_sp_5=0", "d_sp_22=0", "d_oil_1=0", "d_oil_5=0", "d_oil_22=0", "d_term_spread=0", "d_credit_spread=0", "d_volume=0"), white.adjust="hc3"))
print(linearHypothesis(sent.fit.electra, c("d_sp_1=0", "d_sp_5=0", "d_sp_22=0", "d_oil_1=0", "d_oil_5=0", "d_oil_22=0", "d_term_spread=0", "d_credit_spread=0", "d_volume=0"), white.adjust="hc3"))

sent.fit.bert.nw <- coeftest(sent.fit.bert, vcov=NeweyWest(sent.fit.bert, prewhite=F, adjust=T))
print(sent.fit.bert.nw)
sent.fit.roberta.nw <- coeftest(sent.fit.roberta, vcov=NeweyWest(sent.fit.roberta, prewhite=F, adjust=T))
print(sent.fit.roberta.nw)
sent.fit.xlnet.nw <- coeftest(sent.fit.xlnet, vcov=NeweyWest(sent.fit.xlnet, prewhite=F, adjust=T))
print(sent.fit.xlnet.nw)
sent.fit.electra.nw <- coeftest(sent.fit.electra, vcov=NeweyWest(sent.fit.electra, prewhite=F, adjust=T))
print(sent.fit.electra.nw)




df <- cbind(id=rownames(df), df)

#saving fitted values
df.ols_pred.bert <- as.data.frame(sent.fit.bert$fitted.values)
df.ols_pred.bert <- cbind(id=rownames(df.ols_pred.bert), df.ols_pred.bert)
df <- merge(df, df.ols_pred.bert, by=c("id"), all.x=TRUE)
df <- setnames(df, "sent.fit.bert$fitted.values", "fitted_sent_bert")
df$fitted_sent_bert_positive <-ifelse(df$fitted_sent_bert >= 0.5, df$fitted_sent_bert, 0)
df$fitted_sent_bert_negative <- ifelse(df$fitted_sent_bert >= 0.5, 0, 1-df$fitted_sent_bert)
df$fitted_sent_bert_positive[is.na(df$fitted_sent_bert_positive)] <- 0
df$fitted_sent_bert_negative[is.na(df$fitted_sent_bert_negative)] <- 0

df.ols_pred.roberta <- as.data.frame(sent.fit.roberta$fitted.values)
df.ols_pred.roberta <- cbind(id=rownames(df.ols_pred.roberta), df.ols_pred.roberta)
df <- merge(df, df.ols_pred.roberta, by=c("id"), all.x=TRUE)
df <- setnames(df, "sent.fit.roberta$fitted.values", "fitted_sent_roberta")
df$fitted_sent_roberta_positive <-ifelse(df$fitted_sent_roberta >= 0.5, df$fitted_sent_roberta, 0)
df$fitted_sent_roberta_negative <- ifelse(df$fitted_sent_roberta >= 0.5, 0, 1-df$fitted_sent_roberta)
df$fitted_sent_roberta_positive[is.na(df$fitted_sent_roberta_positive)] <- 0
df$fitted_sent_roberta_negative[is.na(df$fitted_sent_roberta_negative)] <- 0

df.ols_pred.xlnet <- as.data.frame(sent.fit.xlnet$fitted.values)
df.ols_pred.xlnet <- cbind(id=rownames(df.ols_pred.xlnet), df.ols_pred.xlnet)
df <- merge(df, df.ols_pred.xlnet, by=c("id"), all.x=TRUE)
df <- setnames(df, "sent.fit.xlnet$fitted.values", "fitted_sent_xlnet")
df$fitted_sent_xlnet_positive <-ifelse(df$fitted_sent_xlnet >= 0.5, df$fitted_sent_xlnet, 0)
df$fitted_sent_xlnet_negative <- ifelse(df$fitted_sent_xlnet >= 0.5, 0, 1-df$fitted_sent_xlnet)
df$fitted_sent_xlnet_positive[is.na(df$fitted_sent_xlnet_positive)] <- 0
df$fitted_sent_xlnet_negative[is.na(df$fitted_sent_xlnet_negative)] <- 0

df.ols_pred.electra <- as.data.frame(sent.fit.electra$fitted.values)
df.ols_pred.electra <- cbind(id=rownames(df.ols_pred.electra), df.ols_pred.electra)
df <- merge(df, df.ols_pred.electra, by=c("id"), all.x=TRUE)
df <- setnames(df, "sent.fit.electra$fitted.values", "fitted_sent_electra")
df$fitted_sent_electra_positive <-ifelse(df$fitted_sent_electra >= 0.5, df$fitted_sent_electra, 0)
df$fitted_sent_electra_negative <- ifelse(df$fitted_sent_electra >= 0.5, 0, 1-df$fitted_sent_electra)
df$fitted_sent_electra_positive[is.na(df$fitted_sent_electra_positive)] <- 0
df$fitted_sent_electra_negative[is.na(df$fitted_sent_electra_negative)] <- 0


#saving residuals
df.ols_res.bert <- as.data.frame(residuals(sent.fit.bert))
df.ols_res.bert <- cbind(id=rownames(df.ols_res.bert), df.ols_res.bert)
df <- merge(df, df.ols_res.bert, by=c("id"), all.x=TRUE)
df <- setnames(df, "residuals(sent.fit.bert)", "fitted_res_bert")

df.ols_res.roberta <- as.data.frame(residuals(sent.fit.roberta))
df.ols_res.roberta <- cbind(id=rownames(df.ols_res.roberta), df.ols_res.roberta)
df <- merge(df, df.ols_res.roberta, by=c("id"), all.x=TRUE)
df <- setnames(df, "residuals(sent.fit.roberta)", "fitted_res_roberta")

df.ols_res.xlnet <- as.data.frame(residuals(sent.fit.xlnet))
df.ols_res.xlnet <- cbind(id=rownames(df.ols_res.xlnet), df.ols_res.xlnet)
df <- merge(df, df.ols_res.xlnet, by=c("id"), all.x=TRUE)
df <- setnames(df, "residuals(sent.fit.xlnet)", "fitted_res_xlnet")

df.ols_res.electra <- as.data.frame(residuals(sent.fit.electra))
df.ols_res.electra <- cbind(id=rownames(df.ols_res.electra), df.ols_res.electra)
df <- merge(df, df.ols_res.electra, by=c("id"), all.x=TRUE)
df <- setnames(df, "residuals(sent.fit.electra)", "fitted_res_electra")




#factor analysis
#df <- cbind(id.fa=rownames(df), df)
df.fa <- df[,c("id", "bert_score_prob", "roberta_score_prob", "xlnet_score_prob", "electra_score_prob")]
df.fa <- df.fa[!is.na(df.fa$bert_score_prob), ]
sent.fa <- factanal(df.fa[,c("bert_score_prob", "roberta_score_prob", "xlnet_score_prob", "electra_score_prob")], factors=1, scores="regression")
df.fa$fa_scores <- sent.fa$scores
df.fa <- df.fa[, c("id", "fa_scores")]
df <- merge(df, df.fa, by=c("id"), all.x=TRUE)
df$fa_scores[is.na(df$fa_scores)] <- 0
print(sent.fa)
#unique
sent.fa$uniquenesses
#common
apply(sent.fa$loadings^2,1,sum)

#Lambda <- sent.fa$loadings
#Psi <- diag(sent.fa$uniquenesses)
#S <- sent.fa$correlation
#Sigma <- Lambda %*% t(Lambda) + Psi
#round(S - Sigma, 6)

#computing averaged factor over k \in l days
l <- c(1, 5, 22)
for (i in 1:dim(df)){
  for (k in l){
    if (i > k){
      temp_score <- 0
      count_speeches <- 0
      for (j in 1:k){
        if (df[i-j, "fa_scores"] != 0){
          count_speeches <- count_speeches + 1
        }
        temp_score = temp_score + df[i-j, "fa_scores"]
      }
      if (count_speeches == 0){
        temp_score <- NA
      } else{
        temp_score <- temp_score / count_speeches
        
        df[i-1, paste("fa_scores_p", k, sep="")] <- ifelse(temp_score >= 0.5, temp_score, 0)
        df[i-1, paste("fa_scores_n", k, sep="")] <- ifelse(temp_score >= 0.5, 0, 1-temp_score)
      }
    }
  }
}
for(k in l){
  df[,paste("fa_scores_p", k, sep="")][is.na(df[,paste("fa_scores_p", k, sep="")])] <- 0
  df[,paste("fa_scores_n", k, sep="")][is.na(df[,paste("fa_scores_n", k, sep="")])] <- 0
}


#creating means of past sentiment values
df$bert_score_prob[is.na(df$bert_score_prob)] <- 0
df$roberta_score_prob[is.na(df$roberta_score_prob)] <- 0
df$xlnet_score_prob[is.na(df$xlnet_score_prob)] <- 0
df$electra_score_prob[is.na(df$electra_score_prob)] <- 0
df$fitted_sent_bert[is.na(df$fitted_sent_bert)] <- 0
df$fitted_res_bert[is.na(df$fitted_res_bert)] <- 0
df$fitted_sent_roberta[is.na(df$fitted_sent_roberta)] <- 0
df$fitted_res_roberta[is.na(df$fitted_res_roberta)] <- 0
df$fitted_sent_xlnet[is.na(df$fitted_sent_xlnet)] <- 0
df$fitted_res_xlnet[is.na(df$fitted_res_xlnet)] <- 0
df$fitted_sent_electra[is.na(df$fitted_sent_electra)] <- 0
df$fitted_res_electra[is.na(df$fitted_res_electra)] <- 0

#computing average (equal weighted) sentiment for each measure.
Q <- c("bert", "roberta", "xlnet", "electra")
l <- c(1, 5, 22)
for (i in 1:dim(df)){
  for (k in l){
    for (q in Q){
      if (i > k){
        vaktemp_sent <- 0
        vaktemp_fitted_sent <- 0
        vaktemp_res_sent <- 0
        count_speeches <- 0
        for (j in 1:k){
          if (df[i-j, paste("fitted_sent_", q, sep="")] != 0){
            count_speeches <- count_speeches + 1
          }
          vaktemp_sent = vaktemp_sent + df[i-j, paste(q, "_score_prob", sep="")]
          vaktemp_fitted_sent = vaktemp_fitted_sent + df[i-j, paste("fitted_sent_", q, sep="")]
          vaktemp_res_sent = vaktemp_res_sent + df[i-j, paste("fitted_res_", q, sep="")]
        }
        if (count_speeches == 0){
          vaktemp_sent <- NA
          vaktemp_fitted_sent <- NA
          vaktemp_res_sent <- NA
        } else{
          vaktemp_sent <- vaktemp_sent / count_speeches
          vaktemp_fitted_sent <- vaktemp_fitted_sent / count_speeches
          vaktemp_res_sent <- vaktemp_res_sent / count_speeches
          
          df[i-1, paste("sent_p_", q, k, sep="")] <- ifelse(vaktemp_sent >= 0.5, vaktemp_sent, 0)
          df[i-1, paste("sent_n_", q, k, sep="")] <- ifelse(vaktemp_sent >= 0.5, 0, 1-vaktemp_sent)
          
          df[i-1, paste("fitted_sent_p_", q, k, sep="")] <- ifelse(vaktemp_fitted_sent >= 0.5, vaktemp_fitted_sent, 0)
          df[i-1, paste("fitted_sent_n_", q, k, sep="")] <- ifelse(vaktemp_fitted_sent >= 0.5, 0, 1-vaktemp_fitted_sent)
          df[i-1, paste("fitted_res_p_", q, k, sep="")] <- ifelse(vaktemp_fitted_sent >= 0.5, vaktemp_res_sent, 0)
          df[i-1, paste("fitted_res_n_", q, k, sep="")] <- ifelse(vaktemp_fitted_sent >= 0.5, 0, 1-vaktemp_res_sent)
        }
      }
    }
  }
}

for (i in 1:dim(df)){
  for (k in l){
    sent_p_ew <- 0
    sent_n_ew <- 0
    sent_fitted_p_ew <- 0
    sent_fitted_n_ew <- 0
    sent_res_p_ew <- 0
    sent_res_n_ew <- 0
    for(q in Q){
      sent_p_ew <- sent_p_ew +  df[i, paste("sent_p_", q, k, sep="")] 
      sent_n_ew <- sent_n_ew +  df[i, paste("sent_n_", q, k, sep="")] 
      sent_fitted_p_ew <- sent_fitted_p_ew +  df[i, paste("fitted_sent_p_", q, k, sep="")] 
      sent_fitted_n_ew <- sent_fitted_n_ew +  df[i, paste("fitted_sent_n_", q, k, sep="")] 
      sent_res_p_ew <- sent_res_p_ew +  df[i, paste("fitted_res_p_", q, k, sep="")] 
      sent_res_n_ew <- sent_res_n_ew +  df[i, paste("fitted_res_n_", q, k, sep="")] 
    }
    df[i, paste("sent_ew_p_", k, sep="")] <- sent_p_ew / length(Q)
    df[i, paste("sent_ew_n_", k, sep="")] <- sent_n_ew / length(Q)
    df[i, paste("fitted_sent_ew_p_", k, sep="")] <- sent_fitted_p_ew / length(Q)
    df[i, paste("fitted_sent_ew_n_", k, sep="")] <- sent_fitted_n_ew / length(Q)
    df[i, paste("fitted_res_ew_p_", k, sep="")] <- sent_res_p_ew / length(Q)
    df[i, paste("fitted_res_ew_n_", k, sep="")] <- sent_res_n_ew / length(Q)
  }
}

#replacing NAs with zeroes in sentiment regressors
for (q in Q){
  for (k in l){
    df[, paste("sent_p_", q, k, sep="")][is.na(df[, paste("sent_p_", q, k, sep="")])] <- 0
    df[, paste("sent_n_", q, k, sep="")][is.na(df[, paste("sent_n_", q,  k, sep="")])] <- 0
    df[, paste("fitted_sent_p_", q, k, sep="")][is.na(df[, paste("fitted_sent_p_", q, k, sep="")])] <- 0
    df[, paste("fitted_sent_n_", q, k, sep="")][is.na(df[, paste("fitted_sent_n_", q, k, sep="")])] <- 0
    df[, paste("fitted_res_p_", q, k, sep="")][is.na(df[, paste("fitted_res_p_", q, k, sep="")])] <- 0
    df[, paste("fitted_res_n_", q, k, sep="")][is.na(df[, paste("fitted_res_n_", q, k, sep="")])] <- 0
    df[, paste("sent_ew_p_", k, sep="")][is.na(df[, paste("sent_ew_p_", k, sep="")])] <- 0
    df[, paste("sent_ew_n_", k, sep="")][is.na(df[, paste("sent_ew_n_",k, sep="")])] <- 0
    df[, paste("fitted_sent_ew_p_", k, sep="")][is.na(df[, paste("fitted_sent_ew_p_", k, sep="")])] <- 0
    df[, paste("fitted_sent_ew_n_", k, sep="")][is.na(df[, paste("fitted_sent_ew_n_", k, sep="")])] <- 0
    df[, paste("fitted_res_ew_p_", k, sep="")][is.na(df[, paste("fitted_res_ew_p_", k, sep="")])] <- 0
    df[, paste("fitted_res_ew_n_", k, sep="")][is.na(df[, paste("fitted_res_ew_n_", k, sep="")])] <- 0
    
  }
}

 


#correlation matrix
corr_df = df[,c("bert_score_prob", "roberta_score_prob", "xlnet_score_prob", "electra_score_prob")]
corr_df <- corr_df[!is.na(corr_df$roberta_score_prob), ]
cor(corr_df)



#HAR
#Creating formulas for the HAR, HARX, and HARX-FA
har.formula <- c(y0 ~ y1 + y5 + y22)
for (f in har.formula){
  har.fit <- lm(data=df, formula = f, na.action=na.omit)
}
#HARX
har.econ.formula <- c(y0 ~ y1 + y5 + y22 + d_sp_1 + d_sp_5 + d_sp_22 + d_oil_1 + d_oil_5 + d_oil_22 + d_term_spread + d_credit_spread + d_volume)
for (f in har.econ.formula){
  har.econ.fit <- lm(data=df, formula = f, na.action=na.omit)
}
#HARX-FA
har.econ.fa.formula <- c(y0 ~ y1 + y5 + y22 + d_sp_1 + d_sp_5 + d_sp_22 + d_oil_1 + d_oil_5 + d_oil_22 + d_term_spread + d_credit_spread + d_volume + 
                        fa_scores_p1 + fa_scores_p5 + fa_scores_p22 + fa_scores_n1 + fa_scores_n5 + fa_scores_n22 )
for (f in har.econ.fa.formula){
  har.econ.fa.fit <- lm(data=df, formula = f, na.action=na.omit)
}

# creating vectors of model formulas for IV and SM models to be fed into lm()
formulas.single <- c()
formulas.iv <- c()
har.vars <- c("y1", "y5", "y22")
econ.vars <- c("d_sp_1", "d_sp_5", "d_sp_22", "d_oil_1", "d_oil_5", "d_oil_22" , "d_term_spread", "d_credit_spread", "d_volume")
for(q in Q){
  sent.vars <- c()
  iv.vars <- c()
  for (k in l){
    sent.vars <- c(sent.vars, paste("sent_p_", q, k, sep=""), paste("sent_n_", q, k, sep=""))
    iv.vars <- c(iv.vars, paste("fitted_sent_n_", q, k, sep=""), paste("fitted_sent_p_", q, k, sep=""))
    #iv.vars <- c(iv.vars, paste("fitted_res_n_", q, k, sep=""), paste("fitted_res_p_", q, k, sep=""))
  }
  formulas.single <- c(formulas.single, as.formula(paste("y0~", paste(c(har.vars, econ.vars, sent.vars), collapse="+"))))
  formulas.iv <- c(formulas.iv, as.formula(paste("y0~", paste(c(har.vars, econ.vars, iv.vars), collapse="+"))))
}

for(formula_ in formulas.iv){
  f1.fit <- lm(data=df, formula = formula_, na.action=na.omit)
  fit.nw <- coeftest(f1.fit, vcov=NeweyWest(f1.fit, prewhite=F, adjust=T))
  print(fit.nw)
  print(linearHypothesis(f1.fit, c("d_sp_1=0", "d_sp_5=0", "d_sp_22=0", "d_oil_1=0", "d_oil_5=0", "d_oil_22=0", "d_term_spread=0", "d_credit_spread=0", "d_volume=0"), white.adjust="hc3"))
  print(summary(f1.fit))
  x = x
  }



#creating vector of model fomulas for equal weights
formulas.ew.sm <- c()
formulas.ew.iv <- c()
ew.vars.sm <- c()
ew.vars.iv <- c()
for (k in l){
  ew.vars.sm <- c(ew.vars.sm, paste("sent_ew_p_", k, sep=""), paste("sent_ew_n_", k, sep=""))
  ew.vars.iv <- c(ew.vars.iv, paste("fitted_sent_ew_p_", k, sep=""), paste("fitted_sent_ew_n_", k, sep=""))
  #ew.vars.iv <- c(ew.vars.iv, paste("fitted_res_ew_p_", k, sep=""), paste("fitted_res_ew_n_", k, sep=""))
}
formulas.ew.sm <- c(formulas.ew.sm, as.formula(paste("y0~", paste(c(har.vars, econ.vars, ew.vars.sm), collapse="+"))))
formulas.ew.iv <- c(formulas.ew.iv, as.formula(paste("y0~", paste(c(har.vars, econ.vars, ew.vars.iv), collapse="+"))))



#DM test between each model's forecast
#outputs p-values in latex format
all.formulas <- c(har.formula, har.econ.formula, formulas.single, formulas.ew.sm, formulas.iv, formulas.ew.iv, har.econ.fa.formula)
i <- 0
count= 0
dm.test <- matrix(0, nrow=length(all.formulas), ncol=length(all.formulas))
for (f1 in all.formulas){
  i  <- i + 1
  j <- 0
  #print(f1)
  for (f2 in all.formulas){
    j <- j + 1
    if (j > i){
      f1.fit <- lm(data=df, formula = f1, na.action=na.omit)
      f2.fit <- lm(data=df, formula = f2, na.action=na.omit)
      if(AIC(f1.fit) != AIC(f2.fit)){
        #print(count)
        count = count + 1
        print(f1)
        print(f2)
        dm <- dm.test(residuals(f1.fit), residuals(f2.fit), h=h, power=2)  # DM test
        print(dm)
        dm.test[j,i] <- format(round(dm$p.value, 3), nsmall=3)
        #print(adf.test((residuals(f1.fit))^2 -( residuals(f2.fit))^2))
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
      }
    }
  }
}
dm.test <- cbind(c("HAR", "HARX", "SM-BE", "SM-RO", "SM-XL", "SM-EL", "SM-EW", "IV-BE", "IV-RO", "IV-XL", "IV-EL", "IV-EW", "FA"), dm.test)
dm.test <- dm.test[-c(1),]
dm.test <- dm.test[,-c(ncol(dm.test))]
print(xtable(dm.test, type="latex"), include.rownames=FALSE)



#testing out of sample predictive performance
#output is printed in latex format
oos.test <- matrix(0, nrow=length(all.formulas), ncol=4)
formula_index <- 0
for (f in all.formulas){
  formula_index <- formula_index + 1
  i <- 1500
  n <- 0
  mfe <- 0
  mse <- 0
  mae <- 0
  mape <- 0
  print(f)
  print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
  while(i < nrow(df) - h){
    df_train <- df[1:i,]
    df_test <- df[i + h,]
    fit <- lm(data=df_train, formula = f, na.action=na.omit)
    fit.pred <- predict(fit, df_test)
    mfe <- mfe + fit.pred - df[i+h, "y0"]
    mse <- mse + (fit.pred - df[i+h, "y0"])^2
    mae <- mae + abs(fit.pred - df[i+h, "y0"])
    mape <- mape + abs((fit.pred - df[i+h, "y0"]) / df[i+h, "y0"])
    n <- n + 1
    i <- i + 1
  }
  oos.test[formula_index, 1] <- mfe/n
  oos.test[formula_index, 2] <- mse/n
  oos.test[formula_index, 3] <- mae/n
  oos.test[formula_index, 4] <- 100*mape/n
  print("mfe")
  print(mfe/n)
  print("mse")
  print(mse/n)
  #print("rmse")
  #print((mse/n)^0.5)
  print("mae")
  print(mae/n)
  print("mape")
  print(100*mape/n)
}
print(xtable(oos.test, type="latex", digits=5))












#sum speech count by person
aggregate(df$is_speech, by=list(df$author), FUN=sum)

#summary stats
vars <- c("y0", "VIX_nolog", "volume", "tres3m", "tres10y", "term_spread",  "oil", "credit")
for (var in vars){
  print(var)
  print(stat.desc(df[,var]))
}
df.onlyspeech <- df[!is.na(df$speech),]
vars <- c("bert_score_prob", "roberta_score_prob", "xlnet_score_prob", "electra_score_prob")
for (var in vars){
  print(var)
  print(format(stat.desc(df.onlyspeech[,var]), scientific=FALSE))
  print(kurtosis(df.onlyspeech[,var]))
  print(skewness(df.onlyspeech[,var]))
}

#graphs -------------------------------------------------------------------------------------------------------------


#VIX
ggplot(df, aes(x=DATE, y=VIX_nolog)) + 
  geom_line(color="#0a6dc9") + 
  labs(y="VIX", x="") +
  scale_x_date(date_labels="%Y", date_breaks="1 year") + 
  theme_bw() 


#speakers by year
df_reduced_speech <- df[!is.na(df$speech), ]
df_reduced_speech$year <- substring(df_reduced_speech$DATE, 1, 4)

ggplot(df_reduced_speech, aes(year)) +
  geom_bar(fill="#0a6dc9") + 
  labs(y="Number of Speeches", x="") +
  theme_bw()










