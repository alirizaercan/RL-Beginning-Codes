import re
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import CategoricalDistribution


class QwenActorCriticPolicy(ActorCriticPolicy):    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        qwen_model_path: str = "./models/hf_model_ep_3500",
        *args,
        **kwargs,
    ):
        self.qwen_model_path = qwen_model_path
        self.qwen_model = None
        self.qwen_tokenizer = None
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        
        self._load_qwen_model()
    
    def _load_qwen_model(self):
        print(f"Loading Qwen model from {self.qwen_model_path}...")
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_model_path, 
            trust_remote_code=True
        )
        if self.qwen_tokenizer.pad_token is None:
            self.qwen_tokenizer.pad_token = self.qwen_tokenizer.eos_token
        
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            self.qwen_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        
        for param in self.qwen_model.parameters():
            param.requires_grad = True
        self.qwen_model.train()
        
        print("Qwen model loaded and ready for FINE-TUNING!")
        print(f"Trainable parameters: {sum(p.numel() for p in self.qwen_model.parameters() if p.requires_grad):,}")
    
    def _build_mlp_extractor(self):
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                )
                self.latent_dim_pi = output_dim  
                self.latent_dim_vf = output_dim  
            
            def forward(self, features):
                return self.network(features), self.network(features)
            
            def forward_actor(self, features):
                return self.network(features)
            
            def forward_critic(self, features):
                return self.network(features)
        
        self.mlp_extractor = SimpleMLP(
            input_dim=self.observation_space.shape[0],
            output_dim=64
        )
        self.value_net = nn.Linear(64, 1)
    
    def obs_to_prompt(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        x, y, vx, vy, angle, ang_vel, leg1, leg2 = obs[0].cpu().numpy()
        
        instruction = (
            "Return exactly one line in this format:\n"
            "Action: <0|1|2|3>\n\n"
            f"State: [x={x:.4f}, y={y:.4f}, vx={vx:.4f}, vy={vy:.4f}, "
            f"angle={angle:.4f}, angular_vel={ang_vel:.4f}, "
            f"left_leg={leg1:.4f}, right_leg={leg2:.4f}]. "
            f"What action should the lander take?"
        )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )
    
    def get_qwen_action_logits(self, obs):
        batch_size = obs.shape[0]
        all_logits = []
        
        for i in range(batch_size):
            prompt = self.obs_to_prompt(obs[i:i+1])
            
            inputs = self.qwen_tokenizer(
                prompt, 
                return_tensors="pt", 
                add_special_tokens=True
            ).to(obs.device)
            
            outputs = self.qwen_model(**inputs)
            
            last_token_logits = outputs.logits[0, -1, :]
            
            action_token_ids = [
                self.qwen_tokenizer.encode("0", add_special_tokens=False)[0],
                self.qwen_tokenizer.encode("1", add_special_tokens=False)[0],
                self.qwen_tokenizer.encode("2", add_special_tokens=False)[0],
                self.qwen_tokenizer.encode("3", add_special_tokens=False)[0],
            ]
            
            action_logits = torch.stack([last_token_logits[tid] for tid in action_token_ids])
            all_logits.append(action_logits)
        
        return torch.stack(all_logits)
    
    def forward(self, obs, deterministic: bool = False):
        action_logits = self.get_qwen_action_logits(obs)
        
        # Critic için features
        _, latent_vf = self.mlp_extractor(obs)
        values = self.value_net(latent_vf)
        
        distribution = CategoricalDistribution(self.action_space.n)
        distribution.proba_distribution(action_logits)
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob
    
    def evaluate_actions(self, obs, actions):
        action_logits = self.get_qwen_action_logits(obs)
        
        # Critic için features
        _, latent_vf = self.mlp_extractor(obs)
        values = self.value_net(latent_vf)
        
        distribution = CategoricalDistribution(self.action_space.n)
        distribution.proba_distribution(action_logits)
        
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy
    
    def predict_values(self, obs):
        _, latent_vf = self.mlp_extractor(obs)
        return self.value_net(latent_vf)
    
    def _predict(self, observation, deterministic: bool = False):
        actions, _, _ = self.forward(observation, deterministic)
        return actions
