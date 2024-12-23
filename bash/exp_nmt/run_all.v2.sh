# for N in  32 64 96 128 160 192 224 256 ; do
#     nohup bash run_generate.api.mcts.pre_load.sh $N > log.mcts &
#     wait 
#     # nohup bash run_generate.api.prs.pre_load.sh lla-3.1-70b $((N * 5)) > log.single & 
#     # nohup bash run_generate.api.prs.pre_load.sh Mis-large-2407 $((N * 5)) > log.single &
#     # nohup bash run_generate.api.prs.pre_load.sh Mix-8x22B $((N * 5)) > log.single &
#     # nohup bash run_generate.api.prs.pre_load.sh Qw2-72B $((N * 5)) > log.single &
#     # nohup bash run_generate.api.prs.pre_load.sh Wiza-8x22B $((N * 5)) > log.single &
#     # wait

#     #echo $((N * 5))

# done;
# nohup bash run_generate.api.mcts.pre_load.sh 64 > log.mcts & wait
# nohup bash run_generate.api.mcts.pre_load.sh 96 > log.mcts & wait
# nohup bash run_generate.api.mcts.pre_load.sh 128 > log.mcts & wait
# nohup bash run_generate.api.mcts.pre_load.sh 160 > log.mcts & wait
# nohup bash run_generate.api.mcts.pre_load.sh 192 > log.mcts & wait
# nohup bash run_generate.api.mcts.pre_load.sh 224 > log.mcts & wait
# nohup bash run_generate.api.mcts.pre_load.sh 256 > log.mcts & wait




#nohup bash run_generate.api.mcts.pre_load.v2.sh 16 200 > log.mcts & 
#nohup bash run_generate.api.mcts.pre_load.v2.sh 32 200 > log.mcts & wait
#nohup bash run_generate.api.mcts.pre_load.v2.sh 96 200 > log.mcts & 
#nohup bash run_generate.api.mcts.pre_load.v2.sh 128 200 > log.mcts & wait
nohup bash run_generate.api.mcts.pre_load.v2.sh 160 200 > log.mcts & 
#nohup bash run_generate.api.mcts.pre_load.v2.sh 192 200 > log.mcts & wait

#nohup bash run_generate.api.mcts.pre_load.v2.sh 224 200 > log.mcts & 
#nohup bash run_generate.api.mcts.pre_load.v2.sh 256 200 > log.mcts & wait


# server_configs_lla-3.1-70b
# server_configs_Mis-large-2407
# server_configs_Mix-8x22B
# server_configs_Qw2-72B

# server_configs_Wiza-8x22B


# for N in  256 ; do
#     nohup bash run_generate.api.moa.pre_load.sh $N > log.moa &
#     wait
#     nohup bash run_generate.api.ensemble.pre_load.sh $N > log.ensemble &
#     wait   
#     nohup bash run_generate.api.ensemble_seq.pre_load.sh $N > log.ensemble_seq &
#     wait
#     nohup bash run_generate.api.single.sh lla-3.1-70b $((N * 5)) > log.single & 
#     nohup bash run_generate.api.single.sh Mis-large-2407 $((N * 5)) > log.single &
#     nohup bash run_generate.api.single.sh Mix-8x22B $((N * 5)) > log.single &
#     nohup bash run_generate.api.single.sh Qw2-72B $((N * 5)) > log.single &
#     nohup bash run_generate.api.single.sh Wiza-8x22B $((N * 5)) > log.single &
#     wait

#     # wait
#     # nohup bash run_generate.api.prs.pre_load.sh lla-3.1-70b $((N * 5)) > log.single & 
#     # nohup bash run_generate.api.prs.pre_load.sh Mis-large-2407 $((N * 5)) > log.single &
#     # nohup bash run_generate.api.prs.pre_load.sh Mix-8x22B $((N * 5)) > log.single &
#     # nohup bash run_generate.api.prs.pre_load.sh Qw2-72B $((N * 5)) > log.single &
#     # nohup bash run_generate.api.prs.pre_load.sh Wiza-8x22B $((N * 5)) > log.single &
#     # wait

#     #echo $((N * 5))
# done;
