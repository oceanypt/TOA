# 计算每个文件的行数
total_lines=$(wc -l < train.ultrafeedback.part_5.num=19876.jsonl)
lines_per_file=$((total_lines / 5))

# 使用split命令进行分割
split -l $lines_per_file train.ultrafeedback.part_5.num=19876.jsonl train.ultrafeedback.part_5.num=19876










