"""
This file contains a completed implementation of gerbil_connect.server_template for evaluating SpEL.
You can run this file with the following arguments:
    -----------------------------------------------------------------------------
    To evaluate SpEL with no mention-specific candidates, run:
        python server.py bert False False
    To evaluate SpEL with KB+Yago candidate sets, run:
        python server.py bert True True
    To evaluate SpEL with context-agnostic PPRforNED candidate sets, run:
        python server.py bert True False True
    To evaluate SpEL with context-aware PPRforNED candidate sets, run:
        python server.py bert True False False
    -----------------------------------------------------------------------------
    To replicate our OpenAI-GPT-3.5 experiments, run:
        python server.py openai False False False
    -----------------------------------------------------------------------------
"""
import sys
import os
import json
from threading import Lock
from flask import Flask, request
from flask_cors import CORS, cross_origin
from gerbil_connect.nif_parser import NIFParser
from spel.configuration import device, get_n3_entity_to_kb_mappings
from spel.candidate_manager import CandidateManager

app = Flask(__name__, static_url_path='', static_folder='../../../frontend/build')
cors = CORS(app, resources={r"/suggest": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

app.add_url_rule('/', 'root', lambda: app.send_static_file('index.html'))

gerbil_communication_done = False
gerbil_query_count = 0
annotator = None
annotate_result = []
candidates_manager_to_use = None
n3_entity_to_kb_mappings = None

lock = Lock()
assert len(sys.argv) >= 3, "Run the script with the expected annotator_name and use_candidates!"
annotator_name = sys.argv[1]
if annotator_name == 'bert':
    from spel.evaluate_local import SpELEvaluator
    from spel.data_loader import dl_sa
    annotator_class = SpELEvaluator
elif annotator_name == 'openai':
    from openai_gpt.evaluate_local import GPT3Annotator
    from spel.data_loader import dl_sa
    annotator_class = GPT3Annotator
else:
    raise ValueError(f"Undefined annotator: {annotator_name}")
use_candidates = sys.argv[2][0].lower() == 't'
load_kb_plus_yago = sys.argv[3][0].lower() == 't' if use_candidates else False
is_context_agnostic = sys.argv[4][0].lower() == 't' if not load_kb_plus_yago else False
print(f" * Loading the annotator of type: {annotator_class.__name__}")
if use_candidates and load_kb_plus_yago:
    print(f" * Loading the KB+YAGO candidates!")
    candidates_manager_to_use = CandidateManager(dl_sa.mentions_vocab, is_kb_yago=True, is_ppr_for_ned=False,
                                                 is_context_agnostic=is_context_agnostic, is_indexed_for_spans=False)
    print(" * WARNING! make sure you do not load candidates for OOD datasets!")
elif use_candidates and not load_kb_plus_yago:
    c_agnostic_str = "context agnostic" if is_context_agnostic else "context aware"
    print(f" * Loading the {c_agnostic_str} PPRforNED candidates!")
    candidates_manager_to_use = CandidateManager(dl_sa.mentions_vocab, is_kb_yago=False, is_ppr_for_ned=True,
                                                 is_context_agnostic=is_context_agnostic, is_indexed_for_spans=True)
elif not use_candidates:
    print(f" * (not) loading the candidates!")


def extract_dump_res_json(parsed_collection):
    return {
        "text": parsed_collection.contexts[0].mention,
        "value": [{"start": phrase.beginIndex, "end": phrase.endIndex, "tag": phrase.taIdentRef}
                  for phrase in parsed_collection.contexts[0]._context.phrases]
    }

def generic_annotate(nif_bytes, load_aida_finetuned, kb_prefix):
    global gerbil_communication_done, gerbil_query_count, annotator, candidates_manager_to_use
    parsed_collection = NIFParser(nif_bytes.decode('utf-8').replace('\\@', '@'), format='turtle')
    if gerbil_communication_done:
        gerbil_query_count += 1
        print("Received query number {} from gerbil!".format(gerbil_query_count))
        with lock:
            annotator.annotate(parsed_collection, ignore_non_aida_vocab=load_aida_finetuned, kb_prefix=kb_prefix,
                               candidates_manager=candidates_manager_to_use)
    else:
        print(" * Handshake to Gerbil was successful!")
        annotator = annotator_class()
        annotator.init_model_from_scratch(device=device)
        if load_aida_finetuned:
            annotator.shrink_classification_head_to_aida(device=device)
            annotator.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=3)
        else:
            annotator.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=2)
        gerbil_communication_done = True
        return nif_bytes
    try:
        res = parsed_collection.nif_str(format='turtle')
        res_json = extract_dump_res_json(parsed_collection)
        annotate_result.append(res_json)
        return res
    except Exception:
        return nif_bytes

@app.route('/annotate_aida', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origins='*')
def annotate_aida():
    """Use this API for AIDA dataset."""
    return generic_annotate(request.data, True, "http://en.wikipedia.org/wiki/")

@app.route('/annotate_wiki', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origins='*')
def annotate_wiki():
    """Use this API for MSNBC dataset."""
    return generic_annotate(request.data, False, "http://en.wikipedia.org/wiki/")

@app.route('/annotate_dbpedia', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origins='*')
def annotate_dbpedia():
    """Use this API for OKE, KORE, and Derczynski datasets."""
    return generic_annotate(request.data, False, "http://dbpedia.org/resource/")

@app.route('/annotate_n3', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin(origins='*')
def annotate_n3():
    """Use this API for N3 Evaluation dataset."""
    global n3_entity_to_kb_mappings
    if n3_entity_to_kb_mappings is None:
        n3_entity_to_kb_mappings = get_n3_entity_to_kb_mappings()
    return generic_annotate(request.data, False, n3_entity_to_kb_mappings)

if __name__ == '__main__':
    try:
        app.run(host="localhost", port=int(os.environ.get("PORT", 3002)), debug=False)
    finally:
        if annotate_result:
            with open(f"annotate_{annotator_name}_result.json", "w", encoding="utf-8") as f:
                f.write(json.dumps({"annotations": annotate_result}))