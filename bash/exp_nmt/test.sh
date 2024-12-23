filename="test.cs_to_en.num=200.jsonl"
result=$(echo $filename | cut -d '.' -f 2 | cut -d '_' -f 1)
echo $result
