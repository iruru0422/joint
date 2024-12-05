import tarfile
import os
import json


# 出力するtar.gzファイルのパス
output_tar_gz_path = '/app/spel-aida-model/aida-conll-spel-roberta-tokenized-aug-23-2023-100.tar.gz'

# インデントありのJSONファイルのパス
input_json_path = '/app/spel-aida-model/conll_dataset_20230823_254-20_auto_annotate.json'
# 圧縮するJSONファイルのパス
output_json_file_path = '/app/spel-aida-model/conll_dataset_20230823_254-20.json'

# インデントなしのJSONファイルを作成
with open(input_json_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

with open(output_json_file_path, 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, separators=(',', ':'))

# tar.gzファイルを作成
with tarfile.open(output_tar_gz_path, 'w:gz') as tar:
    # JSONファイルを追加
    tar.add(output_json_file_path, arcname=os.path.basename(output_json_file_path))

print(f"{output_tar_gz_path} にJSONファイルがtar.gz形式で保存されました。")
