#!/bin/bash

for lambda in $(seq 40 10 100); do
    echo "" > score
    for i in $(seq 1 50); do
        java -Dlambda="${lambda}" -Dverbose=false -jar testrun.jar -submission=player28 -evaluation=SchaffersEvaluation -seed=$RANDOM 2>>/dev/null 1>> score
    done
    avgScore=$(grep Score score | awk '{sum += $2; n++} END {print sum / n}')
    echo "${lambda}: ${avgScore}"
done
