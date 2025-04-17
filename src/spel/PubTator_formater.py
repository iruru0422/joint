import  sys
import json
import gzip
from transformers import RobertaTokenizer

dataset_path = "/app/MedMentions/full/data/corpus_pubtator.txt.gz"

def get_pubtator_data():
    """
    Reads the PubTator dataset and returns a list of dictionaries with the following keys:
    - 'pmid': PubMed ID
    - 'text': Text of the article
    - 'annotations': List of annotations, each containing:
        - 'start': Start index of the annotation
        - 'end': End index of the annotation
        - 'label': Label of the annotation
    """

    with gzip.open(dataset_path, 'rt', 'utf-8') as file:
        lines = file.readlines()
        current_pmid = None
        current_text = None
        current_annotations = []
        documents = []

        for line in lines:
            if not line:
                continue

            if '|t|' in line :
                pmid, title_text = line.split('|t|')
                if current_pmid and pmid != current_pmid:
                    documents.append({
                        'pmid': current_pmid,
                        'text': current_text,
                        'annotations': current_annotations
                    })
                    current_pmid = pmid
                    current_annotations = []      

            elif '|a|' in line:
                pmid, abst_text = line.split('|a|')
                current_text = title_text + ' </s> ' + abst_text
                current_text = current_text.replace('\n', '')

            else:
                current_pmid = pmid
                parts = line.split('\t')
                if len(parts) >= 6:
                    start = int(parts[1])
                    end = int(parts[2])
                    mention = parts[3]
                    label = parts[4]
                    cui = parts[5].replace('\n', '')
                    current_annotations.append({
                        'start': start,
                        'end': end,
                        'mention': mention,
                        'label': label,
                        'CUI' : cui
                    })
        # Add the last document
        if current_pmid:
            documents.append({
                'pmid': current_pmid,
                'text': current_text,
                'annotations': current_annotations
            })
    return documents

def set_format(document):
    pmid = document['pmid']
    text = document['text']
    annotations = document['annotations']
    formatted_annotations = []
    
    

def format_pubtator_data(documents):
    """
    Formats the PubTator data into SpEL format.
    """
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    formatted_data = {}
    train_data = []
    valid_data = []
    test_data = []
    splits = ['trng', 'dev', 'test']
    pmids = {}
    for split in splits:
       with open('/app/Medmentions/full/data/corpus_pubtator_pmids_'+ split + '.txt', 'w') as r_pmid:
           if split == 'trng':
               split = 'train'
           elif split == 'dev':
               split = 'valid'
           pmids[split] = r_pmid.readlines()

    for document in documents:
        if document['pmid'] in pmids['train']:
            train_data.append(document)
        elif document['pmid'] in pmids['valid']:
            valid_data.append(document)
        elif document['pmid'] in pmids['test']:
            test_data.append(document)
        


if __name__ == "__main__":
    pubtator_data = get_pubtator_data()