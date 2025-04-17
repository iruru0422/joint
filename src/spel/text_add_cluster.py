import  os
import json
import string
from tqdm import tqdm
import torch
import numpy as np
from transformers import RobertaTokenizer
from spel.data_loader import dl_sa 
from openai import OpenAI
 
client = OpenAI(
    api_key=os.getenv("OPEN_AI_API_KEY")
)

cluster_entity_path = "/app/Test/generate_data/clusters.txt"
with open(cluster_entity_path, 'r') as data_file:
    cluster_entity = []
    current_cluster = []
    for line in data_file:
        stripped_line = line.strip()
        if stripped_line == "":
            if current_cluster:
                cluster_entity.append(current_cluster)
                current_cluster = []
        else:
            current_cluster.append(stripped_line)
    if current_cluster:  # 最後のクラスターを追加
        cluster_entity.append(current_cluster)

# with open(entity_file_path, 'r') as data_file:
#     entity_list = [line.strip() for line in data_file.readlines()]

mention_etoa = dl_sa.aida_canonical_redirects
aliases = dict()
for key, value in mention_etoa.items():
    aliases[value] = key

mentions_vocab, mentions_itos= dl_sa.get_aida_vocab_and_itos()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# generate_txt = "/app/Test/generate_data/generate_data_text.txt"
# generate_entity_list = "/app/Test/generate_data/generate_data_entity.txt"

# with open(generate_txt, 'r') as data_file:
#     text_list = [line.strip() for line in data_file.readlines()]

# with open(generate_entity_list, 'r') as data_file:
#     entity_list = [line.strip().split(", ") for line in data_file.readlines()]

document_list = []
with open('/app/spel-aida-model/conll_dataset_20230823_254-20_origin.json', 'r', encoding='utf-8') as json_file:
    data_json = json.load(json_file)

to_output = []
ejected_text = []
for cluster in tqdm(current_cluster):
    tokens = []
    IDs = []
    BIO_tag = []
    BIO_id = []
    mention = []
    mention_id = []
    mention_head = -1
    add_mention_mask = []
    current_entity = None
    saved_word = None
    NME = 0
    full_mention = False
    token_number = 0
    token_numbers = []
    candidate_dict = dict()
    document_group = []
    check_words = ["At","A","In","On","Of","To","The","And","For","With","As","By","From","Is","Was","Are","Were","Be","Been","But","Do","Does","Did","Has","Have","Had","If","Or","An","That","This","Which","Who","Whom","wWose","Where","When","Why","How","Not","No","Yes","So","Such","Than","Then","Too","Very","More","Most","All","Any","Each","Few","Some"]

    for t in ["train", "valid", "test"]:
        for datas in data_json[t]:
            for data in datas:
                if data[4] not in candidate_dict:
                    candidate_dict[data[4]] = [data[5],data[7]]
    
    info_list = []

    if cluster == "":
        continue
    if entity in aliases:
        info.append(f"{entity}は{aliases[entity]}と同じ意味です.")
        alias= aliases[entity]
        entity = alias
    else:
        info = ""
        alias = None
    try:
        response = client.chat.completions.create(
            model ="gpt-4o-mini-2024-07-18",
            messages=[
                    {
                    "role": "system",
                    "content": 'あなたは言語学者です. 与えられたエンティティを最低5つ使って200token以下の段落状の文章を英語で生成してください.エンティティの意味は()の中の情報に従い，エンティティ中の()の中身は文章には入れず，エンティティリストに入れる．また,段落中に使用した全ての固有表現をリストとして返してください. format: {"content": str, "used_entity": [str]}'
                    },
                    {
                    "role": "user",
                    "content": f"{cluster}\n{info}"
                    }
            ],
                temperature = 0.4,
                max_tokens=250,
        )
        proofeaded_response = client.chat.completions.create(
            model ="gpt-4o-mini-2024-07-18",
            messages=[
                    {
                    "role": "system",
                    "content": 'あなたは文章校正です. 与えられた文章を事実に基づいて修正してください.format: {"content": str, "used_entity": [str]}'
                    },
                    {
                    "role": "user",
                    "content": f"{response.choices[0].message.content}"
                    }
            ],
                temperature = 0.4,
                max_tokens=250,
        )

        parsed_response = proofeaded_response.choices[0].message.content
        parsed_response = json.loads(parsed_response)
        text = parsed_response["content"].replace("_"," ")
        entity_list = parsed_response["used_entity"]

    except KeyError as e:
        print(f"KeyError: The expected key was not found in the response: {e}")
        continue
    except Exception as e:
        print(f"Error occurred while processing the entity '{entity}': {e}")
        continue
 
    for count,word in enumerate(text.split()):
        if saved_word:
            full_word = saved_word + "_" + word
        else:
            full_word = word
        if not any(p in entity for p in [".",",","'","-"]):     
            full_word = full_word.strip(string.punctuation)     
        mention_head = entity.find(full_word.strip(string.punctuation))
        if full_word in check_words:
            mention_head = -1
        if mention_head == 0:
            NME += 1
            current_entity = entity
            if full_word == entity or full_word == alias:
                full_mention = True
            elif full_word.strip(string.punctuation) == entity:
                full_mention = True
                last_punctuation = True
        else:
            current_entity = None
            full_mention = False

        if not current_entity:
            NME = 0
        if count != 0:
            word = " " + word
        tokenized_word = tokenizer.tokenize(word)

        for i,token in enumerate(tokenized_word):
            token_id = tokenizer.convert_tokens_to_ids(token)
            tokens.append(token)
            IDs.append(token_id)
            if (current_entity and not any(punct in current_entity for punct in string.punctuation)) or (len(tokenized_word) == i+1 and full_mention):
                if token in string.punctuation:
                    NME = 0
                    current_entity = None
            if NME == 1:
                BIO_tag.append('B')
                BIO_id.append(0)
                NME += 1
                saved_word = full_word
            elif NME >=2:
                BIO_tag.append('I')
                BIO_id.append(1)
                saved_word = full_word
            else:
                BIO_tag.append('O') 
                BIO_id.append(2)    
                saved_word = None
            mention.append(current_entity)
            token_numbers.append(token_number)
            token_number += 1

        if full_mention:
            saved_word = None
    for token,token_id,bio,bio_id,ment, token_num in zip(tokens,IDs,BIO_tag,BIO_id,mention,token_numbers):
        if ment in candidate_dict:
            document_group.append([token,token_id,bio,bio_id,ment,candidate_dict.get(ment)[0],token_num,candidate_dict.get(ment)[1]])
        else:
            document_group.append([token,token_id,bio,bio_id,ment,-1,token_num,None])
    if all(document_group[i][5] < 0   for i in range(len(document_group))):
        ejected_text.append(text)
        continue
    to_output.append(text)
    document_list.append(document_group)
for document_group in document_list:
    data_json["train"].append(document_group)
with open('/app/spel-aida-model/conll_dataset_20230823_254-20_auto_annotate.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_json, json_file, ensure_ascii=False, indent=4)

with open('/app/Test/generate_data/generated_data4_text.txt', 'w') as f:
    for text in to_output:
        f.write(text.rstrip("\n") + "\n")

with open('/app/Test/generate_data/ejected_text4.txt', 'w') as f:
    for text in ejected_text:
        f.write(text.rstrip("\n") + "\n")