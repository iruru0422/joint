import  os
import json
import string
from tqdm import tqdm
import torch
import numpy as np
from transformers import RobertaTokenizer
from spel.data_loader import dl_sa 
from openai import OpenAI
from transformers import BertTokenizer, BertModel
# from scipy.spatial.distance import cosine

# def get_entity_embedding(entity):
#     input_ids = tokenizer.encode(entity, return_tensors='pt', truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(input_ids)
#     last_hidden_states = outputs.last_hidden_state
#     entity_embedding = last_hidden_states.mean(dim=1).squeeze()
#     return entity_embedding.numpy()

# def cosine_similarity(vec1, vec2):
#     return 1 - cosine(vec1, vec2)


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')


client = OpenAI(
    api_key=os.getenv("OPEN_AI_API_KEY")
)
entity_file_path = "/app/SpEL/resources/vocab/aida_no_entry.txt"
aida_entity_file_path = "/app/SpEL/resources/vocab/aida.txt"
enwiki_entity_file_path = "/app/SpEL/resources/vocab/enwiki_20230827.txt"
out_of_domain_entity_file_path = "/app/SpEL/resources/vocab/out_of_domain.txt"

# cluster_entity_path = "/app/Test/generate_data/clusters.txt"
# with open(cluster_entity_path, 'r') as data_file:
#     cluster_entity = []
#     current_cluster = []
#     for line in data_file:
#         stripped_line = line.strip()
#         if stripped_line == "":
#             if current_cluster:
#                 cluster_entity.append(current_cluster)
#                 current_cluster = []
#         else:
#             current_cluster.append(stripped_line)
#     if current_cluster:  # 最後のクラスターを追加
#         cluster_entity.append(current_cluster)

with open(entity_file_path, 'r') as data_file:
    entity_list = [line.strip() for line in data_file.readlines()]

# 候補生成用
# with open(aida_entity_file_path, 'r') as data_file:
#     all_entity_list = [line.strip() for line in data_file.readlines()]

# with open(enwiki_entity_file_path, 'r') as data_file:
#     for line in data_file.readlines():
#         e = line.strip()
#         if e not in all_entity_list:
#             all_entity_list.append(e)

# with open(out_of_domain_entity_file_path, 'r') as data_file:
#     for line in data_file.readlines():
#         e = line.strip()
#         if e not in all_entity_list:
#             all_entity_list.append(e)

# all_entity_embeddings = [get_entity_embedding(entity) for entity in tqdm(all_entity_list)]
# # 類似度の閾値を設定
# similarity_threshold = 0.8

    
mention_etoa = dl_sa.aida_canonical_redirects
aliases = dict()
for key, value in mention_etoa.items():
    aliases[value] = key

# for entity in entity_list:
#     if entity == "":
#         continue
#     if entity in aliases:
#         info = entity + " means same as " + aliases[entity]
#         alias_list.append(aliases[entity]) 
#     else:
#         info = ""
#         alias_list.append(None)
#     response = client.chat.completion.create(
#     engine="gpt-4o-mini-2024-07-18",
#     messages=[
#             {
#             "role": "system",
#             "content": "You are a language researcher. Please create 1 paragraph to use entity linking using an entity in the provided list in paragraph."
#             },
#             {
#             "role": "user",
#             "content": f"{entity}\n{info}"
#             }
#         ],  
#         response_format = {
#                 "type": "json_schema",
#                 "json_schema": {
#                     "name": "content_and_entity",
#                     "strict": True,
#                     "schema": {
#                         "type": "object",
#                         "properties": {
#                             "content": {
#                                 "type": "string",
#                             },
#                             "used_entity": {
#                                 "type": "array",
#                                 "items": {
#                                     "type": "string"
#                                 }
#                             },
#                         },
#                         "required": ["content","used_entity"],
#                         "additionalProperties": False
#                     }
                    # temperature = 0.1,
#                   max_tokens=100,
#                 }
#             },
#     )
#     text_list.append(response["paragraph"])

mentions_vocab, mentions_itos= dl_sa.get_aida_vocab_and_itos()
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

document_list = []
with open('/app/spel-aida-model/conll_dataset_20230823_254-20_origin.json', 'r', encoding='utf-8') as json_file:
    data_json = json.load(json_file)

to_output = []
ejected_text = []
document_flag = 1
alias = ""
for entity in tqdm(entity_list):
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
                    candidate_dict[data[4]] = data[7]
    
    # entity_embedding = get_entity_embedding(entity)
    # similarities = [(other_entity, cosine_similarity(entity_embedding, other_embedding))
    #                 for other_entity, other_embedding in zip(all_entity_list, all_entity_embeddings)]
    # # 類似度が閾値以上の候補を取得
    # filtered_similarities = [item for item in similarities if item[1] >= similarity_threshold]
    # candidate_dict[entity] = filtered_similarities

    if document_flag:
        if entity == "":
            continue
        if entity in aliases:
            info = entity + "means the same as" + aliases[entity] 
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
                        "content": 'You are a linguist. Generate a coherent English paragraph using all the given entities naturally. The paragraph must adhere to a news-like style appropriate for the AIDA dataset and be no more than 30 tokens long. Ensure that the entities are meaningfully integrated into the context.'
                        },
                        {
                        "role": "user",
                        "content": f"{entity}\n{info}"
                        }
                ],
                    temperature = 0.4,
                    max_tokens=120,
            )
            text = response.choices[0].message.content.replace("_"," ")

        except KeyError as e:
            print(f"KeyError: The expected key was not found in the response: {e}")
            continue
        except Exception as e:
            print(f"Error occurred while processing the entity '{entity}': {e}")
            continue
    
    else:
        text = entity

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
    for token,token_id,bio,bio_id,ment,token_num in zip(tokens,IDs,BIO_tag,BIO_id,mention,token_numbers):
        if ment:
            document_group.append([token,token_id,bio,bio_id,ment,0,token_num,[1]])
        else:
            document_group.append([token,token_id,bio,bio_id,ment,-100,token_num,None])
    if all(document_group[i][5] < 0   for i in range(len(document_group))):
        ejected_text.append(text)
        continue
    to_output.append(text)
    document_list.append(document_group)
for document_group in document_list:
    data_json["train"].append(document_group)
with open('/app/spel-aida-model/conll_dataset_20230823_254-20_auto_annotate.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_json, json_file, ensure_ascii=False, indent=4)

with open('/app/Test/generate_data/generated_data_text_domain2.txt', 'w') as f:
    for text in to_output:
        f.write(text.rstrip("\n") + "\n")

with open('/app/Test/generate_data/ejected_text_domain2.txt', 'w') as f:
    for text in ejected_text:
        f.write(text.rstrip("\n") + "\n")