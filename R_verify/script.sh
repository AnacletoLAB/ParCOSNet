#!/bin/bash

TIMEOUTFILE="time.txt"

#taskset -c 12-23 ../build/COSNet --data /home/alessandro/data/HopfieldNet/tsv_compressedLabels/string.yeast.v10.5.net.n1.tsv --label /home/alessandro/data/HopfieldNet/tsv_compressedLabels/yeast.go.ann.CC.6.june.17.stringID.atl5.tsv --out outYeastCC.tsv --geneOut geneYeastCC.tsv --foldsOut foldsYeastCC.tsv --statesOut statesYeastCC.tsv --nThrd 12 --tttt $TIMEOUTFILE

#taskset -c 12-23 ../build/COSNet --data /home/alessandro/data/HopfieldNet/tsv_compressedLabels/string.human.v10.5.net.n1.tsv --label /home/alessandro/data/HopfieldNet/tsv_compressedLabels/human.go.ann.CC.6.june.17.stringID.atl5.tsv --out outHumanCC.tsv --geneOut geneHumanCC.tsv --foldsOut foldsHumanCC.tsv --statesOut statesHumanCC.tsv --nThrd 12 --tttt $TIMEOUTFILE

#taskset -c 12-23 ../build/COSNet --data /home/alessandro/data/HopfieldNet/tsv_compressedLabels/string.mouse.v10.5.net.n1.tsv --label /home/alessandro/data/HopfieldNet/tsv_compressedLabels/mouse.go.ann.CC.6.june.17.stringID.atl5.tsv --out outMouseCC.tsv --geneOut geneMouseCC.tsv --foldsOut foldsMouseCC.tsv --statesOut statesMouseCC.tsv --nThrd 12 --tttt $TIMEOUTFILE

Rscript verifyResY.R

Rscript verifyResH.R

Rscript verifyResM.R
