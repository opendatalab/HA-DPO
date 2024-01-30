import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from transformers import PreTrainedModel

from .base_dpo_trainer import BaseDPOTrainer

class InstructBLIPDPOTrainer(BaseDPOTrainer):
    
    def dpo_loss(
        self,
        policy_logps,
        ref_logps,
        beta: float,
        reference_free: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        policy_chosen_logps, policy_rejected_logps = policy_logps.chunk(2, dim=0)
        reference_chosen_logps, reference_rejected_logps = ref_logps.chunk(2, dim=0)
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(beta * logits)
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards, \
                policy_chosen_logps, policy_rejected_logps, \
                reference_chosen_logps, reference_rejected_logps
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        image = inputs["image"]
        image = torch.cat([img.unsqueeze(0) for img in image], dim=0)
        question = inputs["prompt"]
        chosen_ans = inputs["chosen"]
        reject_ans = inputs["rejected"]

        inputs_llm, atts_llm = self.model.extract_visual_embeddings(image, question)

        self.model.llm_tokenizer.padding_side = "right"
        self.model.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.model.llm_tokenizer(
            question,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model.max_txt_len,
        ).to(image.device)

        inputs_embeds, attention_mask, targets = self.model.build_inputs_and_targets(
            chosen_ans + reject_ans,
            text_input_tokens,
            inputs_llm,
            atts_llm,
            image.device
        )

        with self.model.maybe_autocast():
            policy_logits = self.model.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            ).logits  # [B, L, V]

            policy_logps = self._get_batch_logps(
                policy_logits,
                targets.long(),
                False
            )

            with torch.no_grad():
                ref_logits = self.ref_model.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                ).logits  # [B, L, V]

                ref_logps = self._get_batch_logps(
                    ref_logits,
                    targets.long(),
                    False
                )
                
        losses, chosen_rewards, rejected_rewards, \
        policy_chosen_logps, policy_rejected_logps, \
        reference_chosen_logps, reference_rejected_logps = self.dpo_loss(
            policy_logps=policy_logps, ref_logps=ref_logps, beta=self.beta, reference_free=False
        )
        loss = losses.mean()
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        prefix, metrics = "train_", {}
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"policy_{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return (loss, metrics)
        return loss
