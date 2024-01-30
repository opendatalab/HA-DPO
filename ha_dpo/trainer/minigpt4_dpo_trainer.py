import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from .base_dpo_trainer import BaseDPOTrainer

class MiniGPT4DPOTrainer(BaseDPOTrainer):
    def concatenated_forward(
        self, model, batch
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        
        concatenated_batch = batch
        len_chosen = batch["concatenated_input_embeds"].shape[0] // 2
        
        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model.llama_model(
            inputs_embeds=concatenated_batch["concatenated_input_embeds"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)
        
        cal_batch_logp = self._get_batch_logps
        all_logps = cal_batch_logp(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # don't count image embeds logits
        loss_mask = batch["concatenated_labels"] != self.label_pad_token_id
        logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        chosen_logits = logits[:len_chosen]
        rejected_logits = logits[len_chosen:]
        chosen_logits = [l.detach().cpu().mean() for l in chosen_logits]
        rejected_logits = [l.detach().cpu().mean() for l in rejected_logits]
        chosen_logits = sum(chosen_logits)/len_chosen
        rejected_logits = sum(rejected_logits)/len_chosen
        
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        model,
        batch,
        ref_batch,
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(self.model, batch)
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, ref_batch)

        policy_rejected_logps = policy_rejected_logps
        reference_rejected_logps = reference_rejected_logps
           
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"policy_{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits

        return losses.mean(), metrics
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        inputs = {k:inputs[k][0] for k in inputs.keys()}
        pair_inputs, llm_inputs = {k:[] for k in inputs.keys()}, {k:[] for k in inputs.keys()}
        for idx in range(len(inputs["data_type"])):
            if inputs["data_type"][idx] == "ccsbualign":
                for k in inputs.keys():
                    llm_inputs[k].append(inputs[k][idx])
            else:
                for k in inputs.keys():
                    pair_inputs[k].append(inputs[k][idx])
        
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
            
        dpo_inputs = self.model.prepare_minigpt4_dpo_inputs(pair_inputs)
        ref_dpo_inputs = self.ref_model.prepare_minigpt4_dpo_inputs(pair_inputs)
            
        loss, metrics = self.get_batch_metrics(model, dpo_inputs, ref_dpo_inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
            
        # auxilary task to maintain stability of training
        if len(llm_inputs["data_type"]) != 0:
            llm_loss = self.model.forward(llm_inputs)
            loss += self.gamma * llm_loss['loss']
            metrics["language_modeling_loss"] = self.gamma * llm_loss['loss']
        
        if return_outputs:
            return (loss, metrics)
        return loss
