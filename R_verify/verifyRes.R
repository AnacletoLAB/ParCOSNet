library(HEMDAG)
origLabs <- get(load("/home/frasca/BMC_SI_BITS17_vanillaCODE/mouse.go.ann.CC.6.june.17.stringID.atl5.rda"))

a <- read.table("/home/alessandro/dev/src/COSNet/build/output.txt",stringsAsFactors=F)
d <- readLines("/home/alessandro/dev/src/COSNet/build/geneNames.txt")

# Preprazione matrice degli score
GOterms <- a[,1]
b <- a[,-1]
rownames(b) <- GOterms

# Preparazione vettore dei nomi dei geni
d <- d[-1]
colnames(b) <- d

# matrice pronta per la valutazione
S = t(b)
origLabs <- origLabs[rownames(S),colnames(S)]

auc <- AUROC.single.over.classes( origLabs, S )
auprc <- AUPRC.single.over.classes( origLabs, S )
