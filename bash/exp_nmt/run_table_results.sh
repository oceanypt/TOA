#for data in  test.is_to_en.num=1000.jsonl  test.cs_to_en.num=1448.jsonl  test.zh_to_en.num=1875.jsonl   ; do
for data in test.de_to_en.num=1984.jsonl  test.ru_to_en.num=2016.jsonl ; do

nohup bash run_generate.api.mcts.pre_load.sh 32 $data > log.mcts & 
nohup bash run_generate.api.moa.pre_load.sh 32 $data > log.moa & wait 
nohup bash run_generate.api.ensemble.pre_load.sh 32 $data > log.ensemble & 
nohup bash run_generate.api.ensemble_seq.pre_load.sh 32 $data > log.ensemble_seq &  wait

nohup bash run_generate.api.prs.pre_load.sh Mix-8x22B 160 $data > log.prs & 
nohup bash run_generate.api.prs.pre_load.sh Mis-large-2407 160 $data > log.prs & 
nohup bash run_generate.api.prs.pre_load.sh lla-3.1-70b 160 $data > log.prs & 
nohup bash run_generate.api.prs.pre_load.sh Wiza-8x22B 160 $data > log.prs & 
nohup bash run_generate.api.prs.pre_load.sh Qw2-72B 160 $data > log.prs &  wait

nohup bash run_generate.api.single.sh Mix-8x22B 160 $data > log.single & 
nohup bash run_generate.api.single.sh Mis-large-2407 160 $data > log.single & 
nohup bash run_generate.api.single.sh lla-3.1-70b 160 $data > log.single & 
nohup bash run_generate.api.single.sh Wiza-8x22B 160 $data > log.single & 
nohup bash run_generate.api.single.sh Qw2-72B 160 $data > log.single & 

wait

done