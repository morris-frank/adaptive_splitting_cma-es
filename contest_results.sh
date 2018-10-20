#!/bin/bash

bentcigar_interval="10 12 14 16 18 20 22 24 26 28" #10 intervals
schaffers_interval="10 20 30 40 50 60 70 80 90 100" #10 intervals
katsuura_interval="50 100 125 150 175 200 225 250 500 1000" #10 intervals

evaluation="BentCigarFunction" # or SchaffersEvaluation or KatsuuraEvaluation

mkdir "contest_results"
cd "contest_results"
mkdir $evaluation
cd $evaluation

for lambda in $bentcigar_interval; do
    echo "" > score
    mkdir $lambda
    cd $lambda
    for i in $(seq 1 30); do
    	random_seed=$RANDOM
    	echo "$lambda"
    	echo "$random_seed"
    	filename="${evaluation}_${lambda}_${i}_${random_seed}"
    	echo "$filename"
        # Set verbose to true
        java -Dlambda="${lambda}" -Dverbose=false -jar testrun.jar -submission=player28 -evaluation=$evaluation -seed=$random_seed 2>>/dev/null 1>> $filename
    done
   	cd ..
done