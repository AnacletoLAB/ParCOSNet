do.perf.single <- function (Y, pred){
    n <- length(Y)
    results <- rep(0, 4)
    names(results) <- c("Acc", "Prec", "Rec", "F");
        TP <- sum(Y==1 & pred==1)
        FP <- sum(Y!=1 & pred==1)
        FN <- sum(Y==1 & pred!=1)
        if(TP + FP > 0) results["Prec"] <- TP/(TP+FP)
        if(TP + FN > 0) results["Rec"] <- TP/(TP+FN)
        if(TP + FP >0 && TP + FN >0)
             results["F"] <- 2*TP/(2*TP + FP + FN)
        results["Acc"] <- 1-(FP+FN)/n
    return(results)
}



library(HEMDAG)
origLabs <- get(load("/home/frasca/BMC_SI_BITS17_vanillaCODE/yeast.go.ann.CC.6.june.17.stringID.atl5.rda"))

a <- read.table("outYeastCC.tsv",stringsAsFactors=F)
foldsFromFile <- read.table("foldsYeastCC.tsv",stringsAsFactors=F)
pred <- read.table("statesYeastCC.tsv",stringsAsFactors=F)
d <- readLines("geneYeastCC.tsv")

# Preprazione matrice degli score
GOterms <- a[,1]
b <- a[,-1]
rownames(b) <- GOterms

# preparazione matrice states
pred <- pred[,-1]
#print(range(pred))
pred[pred <= 0] <- 0
pred[pred > 0] <- 1
#print(range(pred))
rownames(pred) <- GOterms

# preparazioen matrice folds
foldsFromFile <- foldsFromFile[,-1]
rownames(foldsFromFile) <- GOterms

# Preparazione vettore dei nomi dei geni
d <- d[-1]
colnames(b) <- d
colnames(pred) <- d
colnames(foldsFromFile) <- d

# matrice pronta per la valutazione
S = t(b)
P = t(pred)
folds = t(foldsFromFile)
origLabs <- origLabs[rownames(S),colnames(S)]

m <- ncol(S)
auprc <- rep(0, m)
names(auprc) <- colnames(origLabs)
Rec <- Prec <-  F <- auprc

Init <- proc.time();

for(k in 1:m){
    tmpfolds <- as.vector(folds[,k])
	val <- unique(tmpfolds)+1
	#print(range(val))
	for(i in 1:max(val)) {
		currfold  <- which(tmpfolds==i-1)
		#print(length(currfold))
		auprc[k]<- auprc[k] + AUPRC.single.class( origLabs[currfold,k], S[currfold,k] )
		res <- do.perf.single(as.vector(origLabs[currfold,k]), as.vector(P[currfold,k]))
	#	print(res)
		Prec[k] <- Prec[k] + res["Prec"]
		Rec[k] <- Rec[k] + res["Rec"]
		F[k]<- F[k] + res["F"]
	}
}

End <- proc.time();
time <- End[3]-Init[3];
cat("time (s):", time, "\n" )

auprc <- auprc/length(val)
Rec <- Rec/length(val)
Prec <- Prec/length(val)
F <- F/length(val)
#auc <- AUROC.single.over.classes( origLabs, S )
#auprc <- AUPRC.single.over.classes( origLabs, S )
cat("Average results: Prec , Rec, F, AUPRC \n")
cat(mean(Prec), mean(Rec), mean(F), mean(auprc), sep = "\t", "\n")
