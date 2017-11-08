TIMEOUTFILE="time.txt"

for i in 1 4 8 12;
    do
		echo "nthreads: $i" >> $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.yeast.v10.5.net.n1.tsv --label yeast.go.ann.CC.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.yeast.v10.5.net.n1.tsv --label yeast.go.ann.MF.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.yeast.v10.5.net.n1.tsv --label yeast.go.ann.BP.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.mouse.v10.5.net.n1.tsv --label mouse.go.ann.CC.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.mouse.v10.5.net.n1.tsv --label mouse.go.ann.MF.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.mouse.v10.5.net.n1.tsv --label mouse.go.ann.BP.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.human.v10.5.net.n1.tsv --label human.go.ann.CC.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.human.v10.5.net.n1.tsv --label human.go.ann.MF.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
		-a ./COSNetN --data string.human.v10.5.net.n1.tsv --label human.go.ann.BP.6.june.17.stringID.atl5.tsv \
		--gene gene.tsv --out out.tsv \
		--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

		echo "----------------" >> $TIMEOUTFILE
	done

	for i in 1 4 8 12;
	    do
			echo "nthreads: $i" >> $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.yeast.v10.5.net.n1.tsv --label yeast.go.ann.CC.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.yeast.v10.5.net.n1.tsv --label yeast.go.ann.MF.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.yeast.v10.5.net.n1.tsv --label yeast.go.ann.BP.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.mouse.v10.5.net.n1.tsv --label mouse.go.ann.CC.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.mouse.v10.5.net.n1.tsv --label mouse.go.ann.MF.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.mouse.v10.5.net.n1.tsv --label mouse.go.ann.BP.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.human.v10.5.net.n1.tsv --label human.go.ann.CC.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.human.v10.5.net.n1.tsv --label human.go.ann.MF.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			/usr/bin/time -f "elapsed: %E - CPU: %P - max_set: %M kb" -o $TIMEOUTFILE \
			-a ./COSNetE --data string.human.v10.5.net.n1.tsv --label human.go.ann.BP.6.june.17.stringID.atl5.tsv \
			--gene gene.tsv --out out.tsv \
			--nThrd $i --seed 1234 --tttt $TIMEOUTFILE

			echo "----------------" >> $TIMEOUTFILE
		done
