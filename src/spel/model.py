"""
The implementation of the main annotator class from "SpEL: Structured Prediction for Entity Linking"
"""
import os
import re
import pickle
import numpy
from typing import List
from glob import glob
from itertools import chain

from transformers import AutoModelForMaskedLM, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from spel.utils import store_validation_data_wiki, chunk_annotate_and_merge_to_phrase, \
    get_aida_set_phrase_splitted_documents, compare_gold_and_predicted_annotation_documents
from spel.decao_eval import EntityEvaluationScores, InOutMentionEvaluationResult
from spel.span_annotation import SubwordAnnotation
from spel.data_loader import BERT_MODEL_NAME, dl_sa, tokenizer
from spel.configuration import get_checkpoints_dir, get_aida_train_canonical_redirects, get_ood_canonical_redirects, \
    get_logdir_dir, get_exec_run_file

class SpELAnnotator:
    def __init__(self):
        super(SpELAnnotator, self).__init__()
        self.checkpoints_root = get_checkpoints_dir()
        self.logdir = get_logdir_dir()
        self.exec_run_file = get_exec_run_file()

        self.text_chunk_length = 254
        self.text_chunk_overlap = 20

        self.bert_lm = None
        self.number_of_bert_layers = 0
        self.bert_lm_h = 0
        self.out = None
        self.softmax = None
        self.ner_weight = None
        self.el_weight = None

    def init_model_from_scratch(self, base_model=BERT_MODEL_NAME, device="cpu"):
        """
        This is required to be called to load up the base model architecture before loading the fine-tuned checkpoint.
        """
        if base_model:
            self.bert_lm = AutoModelForMaskedLM.from_pretrained(base_model, output_hidden_states=True,
                                                                cache_dir=get_checkpoints_dir() / "hf").to(device)
            self.disable_roberta_lm_head()
            self.number_of_bert_layers = self.bert_lm.config.num_hidden_layers + 1
            self.bert_lm_h = self.bert_lm.config.hidden_size
            self.ner_classification_head = nn.Linear(self.bert_lm_h, 9).to(device)  # NERタスク用の新しい線形層を追加
            self.out = nn.Embedding(num_embeddings=len(dl_sa.mentions_vocab),
                                    embedding_dim=self.bert_lm_h, sparse=True).to(device)
            self.ner_weight = nn.Parameter(torch.ones(1,device=device)) # NERの重み
            self.el_weight = nn.Parameter(torch.ones(1,device=device)) # ELの重み
            self.schedulers = []

            # # NER用に変更する場合
            # self.out = nn.Linear(in_features=self.bert_lm_h, out_features=len(dl_sa.mentions_vocab)).to(device)

            self.softmax = nn.Softmax(dim=-1)

    def shrink_classification_head_to_aida(self, device):
        """
        This will be called in fine-tuning step 3 to shrink the classification head to in-domain data vocabulary.
        """
        aida_mentions_vocab, aida_mentions_itos = dl_sa.get_aida_vocab_and_itos()
        if self.out_module.num_embeddings == len(aida_mentions_vocab):
            return
        current_state_dict = self.out_module.state_dict()
        new_out = nn.Embedding(num_embeddings=len(aida_mentions_vocab),
                               embedding_dim=self.bert_lm_h, sparse=True).to(device)
        new_state_dict = new_out.state_dict()
        for index_new in range(len(aida_mentions_itos)):
            item_new = aida_mentions_itos[index_new]
            assert item_new in dl_sa.mentions_vocab, \
                "the aida fine-tuned mention vocab must be a subset of the original vocab"
            index_current = dl_sa.mentions_vocab[item_new]
            new_state_dict['weight'][index_new] = current_state_dict['weight'][index_current]
        new_out.load_state_dict(new_state_dict, strict=False)
        self.out = new_out.to(device)
        dl_sa.shrink_vocab_to_aida()
        dl_sa.make_vocab_to_ner()
        model_params = sum(p.numel() for p in self.bert_lm.parameters())
        out_params = sum(p.numel() for p in self.out.parameters())
        print(f' * Shrank model to {model_params+out_params} number of parameters ({model_params} parameters '
              f'for the encoder and {out_params} parameters for the classification head)!')
    
    # def shrink_classification_head_to_aida(self, device):
    #     """
    #     This will be called in fine-tuning step 3 to shrink the classification head to in-domain data vocabulary.
    #     """
    #     aida_mentions_vocab, aida_mentions_itos = dl_sa.get_aida_ner_vocab_and_itos()
    #     if self.out.out_features == 10:  # 10に変更
    #         return

    #     current_state_dict = self.out.state_dict()
        
    #     # nn.Linearに変更
    #     new_out = nn.Linear(in_features=self.bert_lm_h, out_features=10).to(device)

    #     # 重みの転送は必要に応じて行うことができます
    #     new_state_dict = new_out.state_dict()

    #     # # 重みの転送を行うロジック
    #     # for index_new in range(min(len(aida_mentions_itos), 10)):  # サイズの調整
    #     #     item_new = aida_mentions_itos[index_new]

    #     #     # assert item_new in dl_sa.mentions_vocab, \
    #     #     #     "the aida fine-tuned mention vocab must be a subset of the original vocab"
    #     #     index_current = dl_sa.mentions_vocab[item_new]
    #     #     new_state_dict['weight'][index_new] = current_state_dict['weight'][index_current]

    #     new_out.load_state_dict(new_state_dict, strict=False)
    #     self.out = new_out.to(device)
    #     dl_sa.shrink_vocab_to_aida()

    #     model_params = sum(p.numel() for p in self.bert_lm.parameters())
    #     out_params = sum(p.numel() for p in self.out.parameters())
    #     print(f' * Shrank model to {model_params + out_params} number of parameters ({model_params} parameters '
    #         f'for the encoder and {out_params} parameters for the classification head)!')

    def freeze_encoder_and_el(self):
        for param in self.bert_lm.parameters():
            param.requires_grad = False  # エンコーダのパラメータを凍結

        for param in self.out.parameters():
            param.requires_grad = False  # EL層のパラメータを凍結

        print("Encoder and EL layer frozen.")
    
    def unfreeze_encoder_and_el(self):
        for param in self.bert_lm.parameters():
            param.requires_grad = True  # エンコーダのパラメータを更新可能にする

        for param in self.out.parameters():
            param.requires_grad = True  # EL層のパラメータを更新可能にする

        print("Encoder and EL layer unfrozen.") 

    def compute_loss(self, ner_loss, el_loss):
            # L1とL2正則化の強さ
        l1_lambda = 0.01
        l2_lambda = 0.01
        
        # 学習可能な重みを用いたロスの計算
        total_loss = self.ner_weight * ner_loss + self.el_weight * el_loss
        
        # L1とL2の正則化項を追加
        l1_reg = l1_lambda * (self.ner_weight.abs().sum() + self.el_weight.abs().sum())
        l2_reg = l2_lambda * (self.ner_weight.pow(2).sum() + self.el_weight.pow(2).sum())
        
        # 正則化を含めた損失
        total_loss += l1_reg + l2_reg
        return total_loss

    def shrink_classification_head_to_ner(self, device):
        """
        This will be called in fine-tuning step 3 to adjust the classification head for NER with 9 labels.
        """
        new_out = nn.Embedding(num_embeddings=10, embedding_dim=self.bert_lm_h, sparse=True).to(device)
        self.out = new_out.to(device)
        model_params = sum(p.numel() for p in self.bert_lm.parameters())
        out_params = sum(p.numel() for p in self.out.parameters())
        print(f' * Adjusted model to {model_params + out_params} number of parameters ({model_params} parameters '
            f'for the encoder and {out_params} parameters for the classification head)!')



    @property
    def current_device(self):
        return self.lm_module.device

    @property
    def lm_module(self):
        return self.bert_lm.module if isinstance(self.bert_lm, nn.DataParallel) or \
                                      isinstance(self.bert_lm, nn.parallel.DistributedDataParallel) else self.bert_lm

    @property
    def out_module(self):
        return self.out.module if isinstance(self.out, nn.DataParallel) or \
                                  isinstance(self.out, nn.parallel.DistributedDataParallel) else self.out

    @staticmethod
    def get_canonical_redirects(limit_to_conll=True):
        return get_aida_train_canonical_redirects() if limit_to_conll else get_ood_canonical_redirects()

    def create_optimizers(self, encoder_lr=5e-5, decoder_lr=0.1, ner_decoder_lr= 5e-5, exclude_parameter_names_regex=None,warmup_steps=0, num_training_steps=0,freeze_steps=0):
        if exclude_parameter_names_regex is not None:
            bert_lm_parameters = list()
            regex = re.compile(exclude_parameter_names_regex)
            for n, p in list(self.lm_module.named_parameters()):
                if not len(regex.findall(n)) > 0:
                    bert_lm_parameters.append(p)
        else:
            bert_lm_parameters = list(self.lm_module.parameters())
        bert_optim = optim.Adam(bert_lm_parameters, lr=encoder_lr)
        if decoder_lr < 1e-323:
            # IMPORTANT! This is a hack since if we don't consider an optimizer for the last layer(e.g. decoder_lr=0.0),
            #  BCEWithLogitsLoss will become unstable and memory will explode.
            decoder_lr = 1e-323
        out_optim = optim.SparseAdam(self.out.parameters(), lr=decoder_lr)
        ner_optim = optim.Adam(self.ner_classification_head.parameters(), lr=ner_decoder_lr)  # NER用のオプティマイザを追加
        weight_optim = optim.Adam([self.ner_weight, self.el_weight], lr=1e-3)  # 重みのオプティマイザを追加
        if freeze_steps > 0:
            unfreeze_total_steps = num_training_steps-freeze_steps
            unfreeze_warmup_steps = unfreeze_total_steps * 0.1
            self.schedulers.append(get_linear_schedule_with_warmup(bert_optim, num_warmup_steps=unfreeze_warmup_steps, num_training_steps=unfreeze_total_steps))
            self.schedulers.append(get_linear_schedule_with_warmup(out_optim, num_warmup_steps=unfreeze_warmup_steps, num_training_steps=unfreeze_total_steps))
        else:
            self.schedulers.append(get_linear_schedule_with_warmup(bert_optim, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps))
            self.schedulers.append(get_linear_schedule_with_warmup(out_optim, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps))
        self.schedulers.append(get_linear_schedule_with_warmup(ner_optim, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps))
        return bert_optim, out_optim, ner_optim, weight_optim

    @staticmethod
    def create_warmup_scheduler(optimizer, warmup_steps):
        """
        Creates a scheduler which increases the :param optimizer: learning rate from 0 to the specified learning rate
            in :param warmup_steps: number of batches.
        You need to call scheduler.step() after optimizer.step() in your code for this scheduler to take effect
        """
        return optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: epoch / warmup_steps if epoch < warmup_steps else 1.0)

    def get_highest_confidence_model_predictions(self, batch_token_ids, topk_per_token=20, topk_from_batch=8196):
        """
        This function will be used for hard negative mining. For a given input batch, it will return
         the `topk_from_batch` mentions which have had model puzzled. In the process, to reduce the computational
          complexity the model will first select `topk_per_token` number of candidates from the vocabulary, and then
           applies the topk selection on it.
        """
        with torch.no_grad():
            logits = self.get_model_raw_logits_inference(batch_token_ids)
            # topk_logit_per_token, topk_eids_per_token = logits.topk(topk_per_token, sorted=False, dim=-1)
            # This is a workaround to the torch.topk bug for large sized tensors
            topk_logit_per_token, topk_eids_per_token = [], []
            for batch_item in logits:
                topk_probs, topk_ids = batch_item.topk(topk_per_token, sorted=False, dim=-1)
                topk_logit_per_token.append(topk_probs)
                topk_eids_per_token.append(topk_ids)
            topk_logit_per_token = torch.stack(topk_logit_per_token, dim=0)
            topk_eids_per_token = torch.stack(topk_eids_per_token, dim=0)
            i = torch.cat(
                [
                    topk_eids_per_token.view(1, -1),
                    torch.zeros(topk_eids_per_token.view(-1).size(), dtype=torch.long,
                                device=topk_eids_per_token.device).view(1, -1),
                ],
                dim=0,
            )
            v = topk_logit_per_token.view(-1)
            st = torch.sparse.FloatTensor(i, v)
            stc = st.coalesce()
            topk_indices = stc._values().sort(descending=True)[1][:topk_from_batch]
            result = stc._indices()[0, topk_indices]

            return result.cpu().tolist()
            # ###########################################################################################

    def annotate_subword_ids(self, subword_ids_list: List, k_for_top_k_to_keep: int, token_offsets=None) \
            -> List[SubwordAnnotation]:
        with torch.no_grad():
            token_ids = torch.LongTensor(subword_ids_list)
            raw_logits, hidden_states = self.get_model_raw_logits_inference(token_ids, return_hidden_states=True)
            ner_raw_logits, ner_hidden_states = self.get_model_raw_logits_inference_ner(token_ids, return_hidden_states=True)
            logits = self.get_model_logits_inference(raw_logits, ner_raw_logits, hidden_states, k_for_top_k_to_keep, token_offsets)
            return logits
    

    def get_model_raw_logits_training(self, token_ids, label_ids, label_probs):
        # label_probs is not used in this function but provided for the classes inheriting SpELAnnotator.
        enc = self.bert_lm(token_ids).hidden_states[-1]
        out = self.out(label_ids)
        logits = enc.matmul(out.transpose(0, 1))
        return logits
    
    def get_model_raw_logits_training_ner(self, token_ids, label_ids):
        enc = self.bert_lm(token_ids).hidden_states[-1]  # BERTの最後の隠れ状態を取得
        logits = self.ner_classification_head(enc)  # NER用の分類ヘッドを通してロジットを取得
        return logits

    


    def get_model_logits_inference(self, raw_logits, ner_raw_logits,hidden_states, k_for_top_k_to_keep, token_offsets=None) \
            -> List[SubwordAnnotation]:
        # hidden_states is not used in this function but provided for the classes inheriting SpELAnnotator.
        logits = self.softmax(raw_logits)
        ner_logits = self.softmax(ner_raw_logits)
        # The following line could possibly cause errors in torch version 1.13.1
        # see https://github.com/pytorch/pytorch/issues/95455 for more information
        top_k_logits, top_k_indices = logits.topk(k_for_top_k_to_keep)
        top_k_logits = top_k_logits.squeeze(0).cpu().tolist()
        top_k_indices = top_k_indices.squeeze(0).cpu().tolist()

        ner_indices = ner_logits.argmax(dim=-1).squeeze(0).cpu().tolist()

        chunk = ["" for _ in top_k_logits] if token_offsets is None else token_offsets
        return [SubwordAnnotation(p, i, x[0], ner_tag=ner_indices[idx]) for idx,(p, i, x) in enumerate(zip(top_k_logits, top_k_indices, chunk))]

    def get_model_raw_logits_inference(self, token_ids, return_hidden_states=False):
        encs = self.lm_module(token_ids.to(self.current_device)).hidden_states
        out = self.out_module.weight
        logits = encs[-1].matmul(out.transpose(0, 1))
        return (logits, encs) if return_hidden_states else logits
    
    def get_model_raw_logits_inference_ner(self, token_ids, return_hidden_states=False):
    # モデルを使用して隠れ状態を取得
        encs = self.lm_module(token_ids.to(self.current_device)).hidden_states
        # 最後の隠れ層の出力を取得
        last_hidden_state = encs[-1]
        # NER用の線形層を使用してlogitsを計算
        logits = self.ner_classification_head(last_hidden_state)
        return (logits, encs) if return_hidden_states else logits

    def evaluate(self, epoch, batch_size, label_size, best_f1, is_training=True, use_retokenized_wikipedia_data=False,
                 potent_score_threshold=0.82):
        self.bert_lm.eval()
        self.out.eval()
        self.ner_classification_head.eval()  # NER用の分類ヘッドを評価モードに変更
        vocab_pad_id = dl_sa.mentions_vocab['<pad>']

        all_words, all_tags, all_y, all_y_hat, all_predicted, all_token_ids, all_ner_tags, all_ner_predicted = [], [], [], [], [], [], [], []
        subword_eval = InOutMentionEvaluationResult(vocab_index_of_o=dl_sa.mentions_vocab['|||O|||'])
        dataset_name = store_validation_data_wiki(
            self.checkpoints_root, batch_size, label_size, is_training=is_training,
            use_retokenized_wikipedia_data=use_retokenized_wikipedia_data)
        
        with torch.no_grad():
            for d_file in tqdm(sorted(glob(os.path.join(self.checkpoints_root, dataset_name, "*")))):
                batch_token_ids, label_ids, label_probs, eval_mask, label_id_to_entity_id_dict, \
                    batch_entity_ids, is_in_mention, _ , batch_ner_tags= pickle.load(open(d_file, "rb"))
                logits = self.get_model_raw_logits_inference(batch_token_ids)
                logits_ner = self.get_model_raw_logits_inference_ner(batch_token_ids)
                logits_ner = torch.softmax(logits_ner, dim=-1)
                subword_eval.update_scores(eval_mask, is_in_mention, logits)
                subword_eval.update_scores_for_ner(eval_mask, batch_ner_tags, logits_ner)  # NER用のスコアを更新
                y_hat = logits.argmax(-1)
                y_hat_ner = logits_ner.argmax(-1)

                tags = list()
                predtags = list()
                y_resolved_list = list()
                y_hat_resolved_list = list()
                token_list = list()
                ner_tags = list()
                ner_predtags = list()

                for batch_id, seq in enumerate(label_probs.max(-1)[1]):
                    for token_id, label_id in enumerate(seq[:-self.text_chunk_overlap]):
                        if eval_mask[batch_id][token_id].item() == 0:
                            y_resolved = vocab_pad_id
                        else:
                            y_resolved = label_ids[label_id].item()
                        y_resolved_list.append(y_resolved)
                        tags.append(dl_sa.mentions_itos[y_resolved])
                        y_hat_resolved = y_hat[batch_id][token_id].item()
                        y_hat_resolved_list.append(y_hat_resolved)
                        predtags.append(dl_sa.mentions_itos[y_hat_resolved])
                        token_list.append(batch_token_ids[batch_id][token_id].item())
                        if eval_mask[batch_id][token_id].item() != 0:  # 評価対象トークンの場合
                            ner_tag = batch_ner_tags[batch_id][token_id].item()
                            ner_pred_tag = y_hat_ner[batch_id][token_id].item()  # 予測されたNERタグ
                            ner_tags.append(dl_sa.ner_itos[ner_tag])  # 実際のNERタグ
                            ner_predtags.append(dl_sa.ner_itos[ner_pred_tag])  # 予測されたNERタグ


                all_y.append(y_resolved_list)
                all_y_hat.append(y_hat_resolved_list)
                all_tags.append(tags)
                all_predicted.append(predtags)
                all_words.append(tokenizer.convert_ids_to_tokens(token_list))
                all_token_ids.append(token_list)
                all_ner_tags.append(ner_tags)
                all_ner_predicted.append(ner_predtags)
                del batch_token_ids, label_ids, label_probs, eval_mask, \
                    label_id_to_entity_id_dict, batch_entity_ids, logits, y_hat

        y_true = numpy.array(list(chain(*all_y)))
        y_pred = numpy.array(list(chain(*all_y_hat)))
        all_token_ids = numpy.array(list(chain(*all_token_ids)))
        y_true_ner = numpy.array(list(chain(*all_ner_tags)))
        y_pred_ner = numpy.array(list(chain(*all_ner_predicted)))

        if epoch == 59:
            # 一致しない箇所のインデックスを取得
            mismatch_indices = numpy.where((y_true != y_pred) & (y_true != vocab_pad_id))
            # 一致しない y_true と y_pred の値を取得
            mismatches = list(zip([dl_sa.mentions_itos[int(idx)] for idx in y_true[mismatch_indices]],
                      [dl_sa.mentions_itos[int(idx)] for idx in y_pred[mismatch_indices]]))

            # 一致しない箇所を出力
            file_path = '/app/Test/generated_data/mismatches.txt'
            with open(file_path,'w') as f:
                for i, (true_val, pred_val) in enumerate(mismatches):
                    f.write(f"mismatch {i}: {true_val} != {pred_val}\n")
                

        num_proposed = len(y_pred[(1 < y_pred) & (all_token_ids > 0)])
        num_correct = (((y_true == y_pred) & (1 < y_true) & (all_token_ids > 0))).astype(int).sum()
        num_gold = len(y_true[(1 < y_true) & (all_token_ids > 0)])
        num_proposed_ner = len([1 for tag, token_id in zip(y_pred_ner, all_token_ids) if tag != 'O' and token_id > 0])
        num_correct_ner = sum([1 for true_tag, pred_tag, token_id in zip(y_true_ner, y_pred_ner, all_token_ids)
                       if true_tag == pred_tag and true_tag != 'O' and token_id > 0])
        num_gold_ner = len([1 for true_tag, token_id in zip(y_true_ner, all_token_ids) if true_tag != 'O' and token_id > 0])


        precision = num_correct / num_proposed if num_proposed > 0.0 else 0.0
        recall = num_correct / num_gold if num_gold > 0.0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        f05 = 1.5 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
        precision_ner = num_correct_ner / num_proposed_ner if num_proposed_ner > 0.0 else 0.0
        recall_ner = num_correct_ner / num_gold_ner if num_gold_ner > 0.0 else 0.0
        f1_ner = 2.0 * precision_ner * recall_ner / (precision_ner + recall_ner) if precision_ner + recall_ner > 0.0 else 0.0
        f05_ner = 1.5 * precision_ner * recall_ner / (precision_ner + recall_ner) if precision_ner + recall_ner > 0.0 else 0.0
        
        if f1 > best_f1:
            print("Saving the best checkpoint ...")
            config = self.prepare_model_checkpoint(epoch)
            fname = self.get_mode_checkpoint_name()
            torch.save(config, f"{fname}.pt")
            print(f"weights were saved to {fname}.pt")
        if precision > potent_score_threshold and recall > potent_score_threshold and is_training:
            print(f"Saving the potent checkpoint with both precision and recall above {potent_score_threshold} ...")
            config = self.prepare_model_checkpoint(epoch)
            try:
                fname = self.get_mode_checkpoint_name()
                torch.save(config, f"{fname}-potent.pt")
                print(f"weights were saved to {fname}-potent.pt")
            except NotImplementedError:
                pass
        self.bert_lm.train()
        self.out.train()
        self.ner_classification_head.train()
        with open(self.exec_run_file, "a+") as exec_file:
            exec_file.write(f"{precision}, {recall}, {f1}, {f05}, {num_proposed}, {num_correct}, {num_gold}, "
                            f"{epoch+1},,\n")
        return precision, recall, f1, f05, num_proposed, num_correct, num_gold , subword_eval ,precision_ner, recall_ner, f1_ner, f05_ner

    def inference_evaluate(self, epoch, best_f1, dataset_name='testa'):
        def normalize_string(s):
            return s.strip().replace('.', '').replace(' ', '').lower()
        self.bert_lm.eval()
        self.out.eval()
        self.ner_classification_head.eval()  # NER用の分類ヘッドを評価モードに変更
        evaluation_results = EntityEvaluationScores(dataset_name)
        gold_documents = get_aida_set_phrase_splitted_documents(dataset_name)
        # ゴールドと予測の文字列を格納するリスト
        gold_strings = []
        predicted_strings = []
        # エラー分類のリストを作成
        span_errors = []
        entity_errors = []
        p_entity_errors = []
        span_and_entity_errors = []
        span_error_count = 0
        entity_error_count = 0
        p_entity_error_count = 0


        for gold_document in tqdm(gold_documents):
            t_sentence = " ".join([x.word_string for x in gold_document])
            predicted_document = chunk_annotate_and_merge_to_phrase(self, t_sentence, k_for_top_k_to_keep=1)
            comparison_results = compare_gold_and_predicted_annotation_documents(gold_document, predicted_document)

            g_md = set((e[1].begin_character, e[1].end_character)
                       for e in comparison_results if e[0].resolved_annotation)
            p_md = set((e[1].begin_character, e[1].end_character)
                       for e in comparison_results if e[1].resolved_annotation)
            g_el = set((e[1].begin_character, e[1].end_character, dl_sa.mentions_itos[e[0].resolved_annotation])
                       for e in comparison_results if e[0].resolved_annotation)
            p_el = set((e[1].begin_character, e[1].end_character, dl_sa.mentions_itos[e[1].resolved_annotation])
                       for e in comparison_results if e[1].resolved_annotation)
            
            if p_el:
                evaluation_results.record_mention_detection_results(p_md, g_md)
                evaluation_results.record_entity_linking_results(p_el, g_el)
            
            

        #     # ゴールドと予測の比較
        #     for g_entity in g_el:
        #         if g_entity not in p_el:  # 不一致がある場合
        #             g_span = (g_entity[0], g_entity[1])
        #             g_entity_text = g_entity[2]
                    
        #             # 予測に同じエンティティがあるか
        #             p_candidate = [p for p in p_el if p[2] == g_entity_text]
                    
        #             if p_candidate:
        #                 # スパンのみが違う場合
        #                 for p_entity in p_candidate:
        #                     if g_span != (p_entity[0], p_entity[1]):
        #                         span_errors.append((g_entity, p_entity))
        #                         span_error_count += 1
        #             else:
        #                 # エンティティ自体が違う場合
        #                 entity_errors.append(g_entity)
        #                 entity_error_count += 1

        #     # 予測のエンティティにゴールドにないものがあれば、逆のエラーも記録
        #     for p_entity in p_el:
        #         if p_entity not in g_el:
        #             p_span = (p_entity[0], p_entity[1])
        #             p_entity_text = p_entity[2]
                    
        #             # ゴールドに同じエンティティがあるか
        #             g_candidate = [g for g in g_el if g[2] == p_entity_text]
                    
        #             if g_candidate:
        #                 # スパンのみが違う場合
        #                 for g_entity in g_candidate:
        #                     if p_span != (g_entity[0], g_entity[1]):
        #                         span_errors.append((g_entity, p_entity))
        #                         span_error_count += 1
        #             else:
        #                 # エンティティ自体が違う場合
        #                 p_entity_errors.append(p_entity)
        #                 p_entity_error_count += 1
        #             for g_entity in g_candidate:
        #                 if p_span != (g_entity[0], g_entity[1]) and p_entity_text != g_entity[2]:
        #                     span_and_entity_errors.append((g_entity, p_entity))



        # if epoch == 50:
        #     with open("/app/Test/EL/EL_error_no_NER", "w") as file:
        #         file.write(f"### スパンの誤り エラー数: {span_error_count}###\n")
        #         for g_entity, p_entity in span_errors:
        #             file.write(f"ゴールド: {g_entity[2]} (スパン: {g_entity[0]}, {g_entity[1]}) | 予測: {p_entity[2]} (スパン: {p_entity[0]}, {p_entity[1]})\n")

        #         # エンティティエラーの出力
        #         file.write(f"\n### ゴールドエンティティの誤り エラー数: {entity_error_count}###\n")
        #         for g_entity in entity_errors:
        #             file.write(f"ゴールド: {g_entity[2]} (スパン: {g_entity[0]}, {g_entity[1]})\n")

        #         file.write(f"\n### 予測エンティティの誤り エラー数: {p_entity_error_count}###\n")
        #         for p_entity in p_entity_errors:
        #             file.write(f"予測: {p_entity[2]} (スパン: {p_entity[0]}, {p_entity[1]})\n")

        #         # スパンとエンティティエラーの出力
        #         file.write("\n### スパンとエンティティの誤り ###\n")
        #         for g_entity, p_entity in span_and_entity_errors:
        #             file.write(f"ゴールド: {g_entity[2]} (スパン: {g_entity[0]}, {g_entity[1]}) | 予測: {p_entity[2]} (スパン: {p_entity[0]}, {p_entity[1]})\n")

        if evaluation_results.micro_entity_linking.f1.compute() > best_f1:
            print("Saving the best checkpoint ...")
            config = self.prepare_model_checkpoint(epoch)
            fname = self.get_mode_checkpoint_name()
            torch.save(config, f"{fname}.pt")
            print(f"weights were saved to {fname}.pt")
        self.bert_lm.train()
        self.out.train()
        self.ner_classification_head.train()
        return evaluation_results

    def prepare_model_checkpoint(self, epoch):
        chk_point = {
            "bert_lm": self.lm_module.state_dict(),
            "number_of_bert_layers": self.number_of_bert_layers,
            "bert_lm_h": self.bert_lm_h,
            "out": self.out_module.state_dict(),
            "epoch": epoch,
        }
        sub_model_specific_checkpoint_data = self.sub_model_specific_checkpoint_data()
        for key in sub_model_specific_checkpoint_data:
            assert key not in ["bert_lm", "number_of_bert_layers", "bert_lm_h", "out", "epoch"], \
                f"{key} is already considered in prepare_model_checkpoint function"
            chk_point[key] = sub_model_specific_checkpoint_data[key]
        return chk_point

    def disable_roberta_lm_head(self):
        assert self.bert_lm is not None
        self.bert_lm.lm_head.layer_norm.bias.requires_grad = False
        self.bert_lm.lm_head.layer_norm.weight.requires_grad = False
        self.bert_lm.lm_head.dense.bias.requires_grad = False
        self.bert_lm.lm_head.dense.weight.requires_grad = False
        self.bert_lm.lm_head.decoder.bias.requires_grad = False

    def _load_from_checkpoint_object(self, checkpoint, device="cpu"):
        torch.cuda.empty_cache()
        self.bert_lm.load_state_dict(checkpoint["bert_lm"], strict=False)
        self.bert_lm.to(device)
        self.disable_roberta_lm_head()
        self.out.load_state_dict(checkpoint["out"], strict=False)
        self.out.to(device)
        self.number_of_bert_layers = checkpoint["number_of_bert_layers"]
        self.bert_lm_h = checkpoint["bert_lm_h"]
        self.sub_model_specific_load_checkpoint_data(checkpoint)
        self.bert_lm.eval()
        self.out.eval()
        model_params = sum(p.numel() for p in self.bert_lm.parameters())
        out_params = sum(p.numel() for p in self.out.parameters())
        print(f' * Loaded model with {model_params+out_params} number of parameters ({model_params} parameters '
              f'for the encoder and {out_params} parameters for the classification head)!')

    @staticmethod
    def download_from_torch_hub(finetuned_after_step=1):
        assert 4 >= finetuned_after_step >= 1
        if finetuned_after_step == 4:
            # This model is the same SpEL finetuned model after step 3 except that its classification layer projects to
            # the entirety of the step-2 model rather than shrinking it in size
            file_name = "spel-base-step-3-500K.pt"
            # Downloads and returns the finetuned model checkpoint created on Oct-03-2023
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/8nw5fFXdz2yBP5z/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        elif finetuned_after_step == 3:
            file_name = "spel-base-step-3.pt"
            # Downloads and returns the finetuned model checkpoint created on Sep-26-2023 with P=92.06|R=91.93|F1=91.99
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/HpQ3PMm6A3y1NBl/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        elif finetuned_after_step == 2:
            file_name = 'spel-base-step-2.pt'
            # Downloads and returns the pretrained model checkpoint created on Sep-26-2023 with P=77.60|R=77.91|F1=77.75
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/Hf37vc1foluHPBh/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        else:
            file_name = 'spel-base-step-1.pt'
            # Downloads and returns the pretrained model checkpoint created on Sep-11-2023 with P=82.50|R=83.16|F1=82.83
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/9OAoAG5eYeREE9V/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        print(f" * Loaded pretrained model checkpoint: {file_name}")
        return checkpoint

    @staticmethod
    def download_large_from_torch_hub(finetuned_after_step=1):
        assert 4 >= finetuned_after_step >= 1
        if finetuned_after_step == 4:
            # This model is the same SpEL finetuned model after step 3 except that its classification layer projects to
            # the entirety of the step-2 model rather than shrinking it in size
            file_name = "spel-large-step-3-500K.pt"
            # Downloads and returns the finetuned model checkpoint created on Oct-03-2023
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/BCvputD1ByAvILC/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        elif finetuned_after_step == 3:
            file_name = "spel-large-step-3.pt"
            # Downloads and returns the finetuned model checkpoint created on Oct-02-2023 with P=92.53|R=92.99|F1=93.76
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/kBBlYVM4Tr59P0q/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        elif finetuned_after_step == 2:
            file_name = 'spel-large-step-2.pt'
            # Downloads and returns the pretrained model checkpoint created on Oct-02-2023 with P=77.36|R=73.11|F1=75.18
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/rnDiuKns7gzADyb/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        else:
            file_name = 'spel-large-step-1.pt'
            # Downloads and returns the pretrained model checkpoint created on Sep-11-2023 with P=84.02|R=82.74|F1=83.37
            checkpoint = torch.hub.load_state_dict_from_url('https://vault.sfu.ca/index.php/s/bTp6UN2xL7Yh52w/download',
                                                            model_dir=str(get_checkpoints_dir()), map_location="cpu",
                                                            file_name=file_name)
        print(f" * Loaded pretrained model checkpoint: {file_name}")
        return checkpoint


    def load_checkpoint(self, checkpoint_name, device="cpu", rank=0, load_from_torch_hub=False, finetuned_after_step=1):
        if load_from_torch_hub and BERT_MODEL_NAME == "roberta-large":
            checkpoint = self.download_large_from_torch_hub(finetuned_after_step)
            self._load_from_checkpoint_object(checkpoint, device)
        elif load_from_torch_hub and BERT_MODEL_NAME == "roberta-base":
            checkpoint = self.download_from_torch_hub(finetuned_after_step)
            self._load_from_checkpoint_object(checkpoint, device)
        else: # load from the local .checkpoints directory
            if rank == 0:
                print("Loading model checkpoint: {}".format(checkpoint_name))
            fname = os.path.join(self.checkpoints_root, checkpoint_name)
            checkpoint = torch.load(fname, map_location="cpu")
            self._load_from_checkpoint_object(checkpoint, device)

    # #############################FUNCTIONS THAT THE SUB-MODELS MUST REIMPLEMENT####################################
    def sub_model_specific_checkpoint_data(self):
        """
        :return: a dictionary of key values containing everything that matters to the sub-model and is not already
            considered in prepare_model_checkpoint.
        """
        return {}

    def sub_model_specific_load_checkpoint_data(self, checkpoint):
        return

    def get_mode_checkpoint_name(self):
        raise NotImplementedError

    def annotate(self, nif_collection, **kwargs):
        raise NotImplementedError
