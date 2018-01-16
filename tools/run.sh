#/bin/sh
dataset=$1

acc=""
time=""
output=`echo $1 | cut -d . -f 1`
output=${output}"_results.csv"
for j in $(seq 1 $2)
do
    ./bin/gcn $dataset > log.txt
    echo -n $j >> $output
    echo -n " " >> $output
    acc=$(grep Accuracy log.txt | cut -d " " -f 7,10)
    time=$(grep Global log.txt | cut -d " " -f 16)
    echo -n $acc >> $output
    echo -n " " >> $output
    echo $time >> $output
    sleep 5
done

rm log.txt

exit 0
