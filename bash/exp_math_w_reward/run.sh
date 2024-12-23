
for N in  96 160 224  288 ; do

bash run_generate.api.mcts.pre_load.sh $N

bash run_generate.api.ensemble.pre_load.sh $N

done;