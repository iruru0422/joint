"""
The implementation for domain specific fine-tuning, step three. This process is very light and can run on one Nvidia
1060 with 6 GBs of GPU memory.

Running this script will automatically download aida-conll-spel-roberta-tokenized-aug-23-2023.tar.gz (5.1 MBs)
 into /home/<user_name>/.cache/torch/text/datasets/ (in linux systems). The validation set in this dataset will be cached
 the first time the evaluate function is called and the cached data will be stored into .checkpoints named with the
 format: validation_data_cache_b_<batch_size>_l_<label_size>_conll. You do not need to worry about downloading or
 preprocessing the fine-tuning data. The tar.gz data file will not be extracted on your disc.
"""
import os
import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
from tqdm import tqdm

from spel.model import SpELAnnotator
from spel.data_loader import get_dataset
from spel.configuration import device

TRACK_WITH_WANDB = True
if TRACK_WITH_WANDB:
    import wandb

# def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
#     bce_loss = F.cross_entropy(logits, targets, reduction='none')
#     pt = torch.exp(-bce_loss)
#     focal_loss = alpha * (1 - pt) ** gamma * bce_loss
#     return focal_loss.mean()


class FinetuneS3(SpELAnnotator):
    def __init__(self):
        super(FinetuneS3, self).__init__()

    def finetune(self, checkpoint_name, n_epochs, batch_size, bert_dropout=0.2, encoder_lr=5e-5, label_size=8196,
                 accumulate_batch_gradients=4,  exclude_parameter_names_regex='embeddings|encoder\\.layer\\.[0-2]\\.',
                 eval_batch_size=1):
        self.init_model_from_scratch(device=device)
        if checkpoint_name is None:
            self.load_checkpoint(None, device=device, load_from_torch_hub=True, finetuned_after_step=2)
            checkpoint_name = 'enwiki_finetuned_step_2_model_checkpoint'
        else:
            self.load_checkpoint(checkpoint_name, device=device)
        self.shrink_classification_head_to_aida(device=device)
        if label_size > self.out_module.num_embeddings:
            label_size = self.out_module.num_embeddings
        if TRACK_WITH_WANDB:
            wandb.init(
                project="spel-ner-finetune-step-3",
                config={
                    "checkpoint_name": checkpoint_name,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "bert_dropout": bert_dropout,
                    "encoder_lr": encoder_lr,
                    "label_size": label_size,
                    "accumulate_batch_gradients": accumulate_batch_gradients,
                    "exclude_parameter_names_regex": exclude_parameter_names_regex,
                    "eval_batch_size": eval_batch_size
                }
            )
        self.bert_lm.train()
        self.out.train()
        # self.ner_classification_head.train()  # NER用の分類ヘッドも学習モードに設定
        if bert_dropout > 0:
            for m in self.bert_lm.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = bert_dropout
        # freeze_epochs = 0
        # total_steps = 160 //accumulate_batch_gradients * n_epochs
        # warmup_steps = total_steps * 0.1
        # if freeze_epochs > 0:
        #     freeze_steps = 160 //accumulate_batch_gradients * freeze_epochs
        # else:
        #     freeze_steps = 0
        optimizers = self.create_optimizers(encoder_lr, 1e-7, exclude_parameter_names_regex)
        criterion_el = nn.BCEWithLogitsLoss()  # ELタスク用の損失関数
        best_f1 = 0.0
        # self.freeze_encoder_and_el()
        for epoch in range(n_epochs):
            total_loss_el = 0  # ELタスク用の損失
            print(f"Beginning fine-tune epoch {epoch} ...")
            
            _iter_ = tqdm(enumerate(
                get_dataset(dataset_name='aida', split='train', batch_size=batch_size, label_size=label_size,
                            get_labels_with_high_model_score=self.get_highest_confidence_model_predictions)
            ))
            
            cnt_loss = 0
            for iter_, (inputs, subword_mentions) in _iter_:
                # inputs.eval_mask, subword_mentions.dictionary, inputs.raw_mentions are not used!
                subword_mentions_probs = subword_mentions.probs.to(device)
                logits_el = self.get_model_raw_logits_training(
                    inputs.token_ids.to(device), subword_mentions.ids.to(device), subword_mentions_probs)
                logits_el = logits_el.view(-1)  # (N*T, VOCAB)
                label_probs = subword_mentions_probs.view(-1)  # (N*T,)s
                eval_mask = inputs.eval_mask.unsqueeze(-1).expand(-1, -1, 5600).reshape(-1)  # (N*T,)
                
                masked_logits_el = logits_el[eval_mask == 1]
                masked_label_probs = label_probs[eval_mask == 1]

                # # NERタスク用のロジット計算
                # logits_ner = self.get_model_raw_logits_training_ner(
                #     inputs.token_ids.to(device), inputs.ner_tags.to(device)  # NERではlabel_probsは使用しないためNoneを渡す
                # )
                # logits_ner = logits_ner.view(-1, logits_ner.size(-1))  # (N*T, VOCAB)
                
                # ELタスクとNERタスクの損失を別々に計算
                loss_el = criterion_el(masked_logits_el, masked_label_probs.clone().detach().to(device))  # ELラベルを使用                            
                # loss_ner = focal_loss(logits_ner,inputs.ner_tags.view(-1).clone().detach().to(device))  # NERラベルを使用


                total_loss_el += loss_el.detach().item()
                # total_loss_ner += loss_ner.detach().item()
                cnt_loss += 1.0

                el_weight = 1.0
                # ner_weight = 0.0



                # 合計損失をバックプロパゲーション
                loss = loss_el * el_weight
                #loss = self.compute_loss(loss_el, loss_ner)
                loss.backward()

                # loss = criterion_el(logits_el, label_probs)
                # total_loss += loss.detach().item()
                # cnt_loss += 1.0
                # loss.backward()

                if (iter_ + 1) % accumulate_batch_gradients == 0:
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    # if epoch >= freeze_epochs:
                    #     for sceduler in self.schedulers:
                    #         sceduler.step()
                    # else:
                    #     self.schedulers[2].step()
                            

                del logits_el
                del loss_el
                del loss
                del inputs, subword_mentions, label_probs, subword_mentions_probs
                # _iter_.set_description(f"Avg Loss: {total_loss/cnt_loss:.7f}")
                _iter_.set_description(f"Avg EL Loss: {total_loss_el/cnt_loss:.7f}")
                if TRACK_WITH_WANDB and iter_ % 50 == 49:
                    wandb.log({"el/avg_loss": total_loss_el/cnt_loss
                               })   

            print(f"\nEvaluating at the end of epoch {epoch}")

            sprecision, srecall, sf1, sf05, snum_proposed, snum_correct, snum_gold, subword_eval = self.evaluate(
                epoch, eval_batch_size, label_size, 1.1, is_training=False)
            inference_results = self.inference_evaluate(epoch, best_f1)
            if1 = inference_results.micro_entity_linking.f1.compute()
            if best_f1 < if1:
                best_f1 = if1
            print(f"Subword-level evaluation results: precision={sprecision:.5f}, recall={srecall:.5f}, f1={sf1:.5f}, "
                  f"f05={sf05:.5f}, \n num_proposed={snum_proposed}, num_correct={snum_correct}, num_gold={snum_gold}")
            print("Entity-level evaluation results:")
            print(inference_results)
            mm_alloc = torch.cuda.memory_allocated() / (math.pow(2, 30))
            print(f"Current allocated memory: {mm_alloc:.4f} GB")


            if TRACK_WITH_WANDB:
                wandb.log({
                    "el/s_precision": sprecision,
                    "el/s_recall": srecall,
                    "el/s_f1": sf1,
                    "el/s_f05": sf05,
                    "el/s_num_proposed": snum_proposed,
                    "el/s_num_correct": snum_correct,
                    "s_num_gold": snum_gold,
                    # "ner/s_prescision": sprecision_ner,
                    # "ner/s_recall": srecall_ner,
                    # "ner/s_f1": sf1_ner,
                    # "ner/s_f05": sf05_ner,
                    "el/micro_f1": inference_results.micro_entity_linking.f1.compute(),
                    "el/macro_f1": inference_results.macro_entity_linking.f1.compute(),
                    "md/micro_f1": inference_results.micro_mention_detection.f1.compute(),
                    "md/macro_f1": inference_results.macro_mention_detection.f1.compute(),
                    "epoch": epoch,
                    "allocated_memory": mm_alloc,
                })

    def get_mode_checkpoint_name(self):
        return os.path.join(self.checkpoints_root, "spel-step-3")




if __name__ == '__main__':
    try:
        b_annotator = FinetuneS3()
        b_annotator.finetune(checkpoint_name=None, n_epochs=60, batch_size=10, bert_dropout=0.2, label_size=10240,
                             eval_batch_size=2)
    finally:
        if TRACK_WITH_WANDB:
            wandb.finish()
