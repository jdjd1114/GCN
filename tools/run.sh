#/bin/sh
dataset=$1

acc=""
output=`echo $1 | cut -d . -f 1`
output=${output}"_results.csv"
for j in $(seq 1 $2)
do
    ./bin/gcn $dataset 2 0 > log.txt
    echo -n $j >> $output
    echo -n " " >> $output
    acc=$(grep Accuracy log.txt | cut -d " " -f 7,9)
    echo -n $acc >> $output
    echo -n " " >> $output
    grep Global log.txt | cut -d " " -f 27 >> $output
    sleep 5
done

rm log.txt

exit 0
