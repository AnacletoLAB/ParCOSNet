#!/bin/bash
 
 TIMEOUTFILE=synthTime.txt
 
 /usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
                -a ./COSNet --data erdos25k.txt --label generatedLabels25k.txt \
                --gene gene25k.tsv --out out25k.tsv \
                --nThrd 1 --seed 1234 --tttt $TIMEOUTFILE
				
 /usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
                -a ./COSNet --data erdos50k.txt --label generatedLabels50k.txt \
                --gene gene50k.tsv --out out50k.tsv \
                --nThrd 1 --seed 1234 --tttt $TIMEOUTFILE
				
 /usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
                -a ./COSNet --data erdos100k.txt --label generatedLabels100k.txt \
                --gene gene100k.tsv --out out100k.tsv \
                --nThrd 1 --seed 1234 --tttt $TIMEOUTFILE
				
 /usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
                -a ./COSNet --data erdos250k.txt --label generatedLabels250k.txt \
                --gene gene250k.tsv --out out250k.tsv \
                --nThrd 1 --seed 1234 --tttt $TIMEOUTFILE
				
 /usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
                -a ./COSNet --data erdos500k.txt --label generatedLabels500k.txt \
                --gene gene500k.tsv --out out500k.tsv \
                --nThrd 1 --seed 1234 --tttt $TIMEOUTFILE

