#!/bin/bash

lambda="200"

for i in $(seq 1 30); do
    rd=$RANDOM
    echo "${i}/30: ${rd}"
    java -Dlambda="${lambda}" -Dverbose=true -jar testrun.jar -submission=player28 -evaluation=GG_F21_Evaluation -seed="${rd}" 2>>/dev/null 1>> "logs/Gallagher-${rd}.csv"
done
