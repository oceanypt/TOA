# model_num=4

# for N in 16 32 48 64 80 96 112 128 ; do

# nohup bash run_generate.api.moa.pre_load.sh $N > log.moa &  wait
# nohup bash run_generate.api.ensemble.pre_load.sh $N > log.ensemble & wait
# nohup bash run_generate.api.ensemble_seq.pre_load.sh $N > log.ensemble_seq & wait
# N_2=$((model_num * N))
# nohup bash run_generate.api.prs.pre_load.sh lla3.1-8b $N_2 > log.prs &
# nohup bash run_generate.api.prs.pre_load.sh mis-7b-v0.2 $N_2 > log.prs &
# nohup bash run_generate.api.prs.pre_load.sh qwen2-7b $N_2 > log.prs &
# nohup bash run_generate.api.prs.pre_load.sh yi-1.5-16k $N_2 > log.prs &
# wait
# nohup bash run_generate.api.single.sh lla3.1-8b $N_2 > log.prs &
# nohup bash run_generate.api.single.sh mis-7b-v0.2 $N_2 > log.prs &
# nohup bash run_generate.api.single.sh qwen2-7b $N_2 > log.prs &
# nohup bash run_generate.api.single.sh yi-1.5-16k $N_2 > log.prs &

# wait
# done


# #1. moa
# model_num=4
# for N in 16 32 48 64 80 96 112 128 ; do
# nohup bash run_generate.api.moa.pre_load.sh $N > log.moa &
# done


# #2. ensemble
# model_num=4
# for N in 16 32 48 64 80 96 112 128 ; do
# nohup bash run_generate.api.ensemble.pre_load.sh $N > log.ensemble &
# sleep 5
# done


#3. ensemble seq
model_num=4
for N in 16 32 48 64 80 96 112 128 ; do
nohup bash run_generate.api.ensemble_seq.pre_load.sh $N > log.ensemble_seq &
done

