import torch
import torch.nn as nn
from typing import Callable
from gymnasium import spaces
from transformers import AutoTokenizer, AutoModelForCausalLM
from stable_baselines3.common.policies import ActorCriticPolicy
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

        object.__setattr__(self, "qwen_model", None)
        object.__setattr__(self, "qwen_tokenizer", None)

        self._load_qwen_model()

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        self.add_module("qwen_lm", self.qwen_model)
    
        self.optimizer = torch.optim.Adam([
            {"params": self.qwen_model.parameters(), "lr": 1e-5},
            {"params": self.value_net.parameters(),  "lr": 1e-4},
        ])

    def _load_qwen_model(self):
        print(f"Loading Qwen model from {self.qwen_model_path}...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        qwen = AutoModelForCausalLM.from_pretrained(
            self.qwen_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qwen = qwen.to(device) 
    
        object.__setattr__(self, "qwen_model", qwen)
        object.__setattr__(self, "qwen_tokenizer", tokenizer)
        object.__setattr__(self, "_qwen_device", device) 

        if hasattr(qwen, "gradient_checkpointing_enable"):
            qwen.gradient_checkpointing_enable()

        for param in qwen.parameters():
            param.requires_grad = True
        qwen.train()

        object.__setattr__(self, "qwen_model", qwen)
        object.__setattr__(self, "qwen_tokenizer", tokenizer)

        print("Qwen model loaded and ready for fine-tuning!")
        print(
            f"Trainable parameters: "
            f"{sum(p.numel() for p in qwen.parameters() if p.requires_grad):,}"
        )

    def _build_mlp_extractor(self):
        qwen_hidden_size = self.qwen_model.config.hidden_size

        class _StubExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_dim_pi = 4               
                self.latent_dim_vf = qwen_hidden_size 

            def forward(self, features):
                return features, features

            def forward_actor(self, features):
                return features

            def forward_critic(self, features):
                return features

        self.mlp_extractor = _StubExtractor()

        self.value_net = nn.Sequential(
            nn.Linear(qwen_hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self.action_net = nn.Identity()

    def obs_to_prompt(self, obs):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        x, y, vx, vy, angle, ang_vel, leg1, leg2 = (
            obs[0].cpu().float().numpy()
        )

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

    def _get_qwen_features(self, obs):
        batch_size = obs.shape[0]
        
        prompts = [self.obs_to_prompt(obs[i:i+1]) for i in range(batch_size)]
        
        inputs = self.qwen_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,           
            truncation=True,
            add_special_tokens=True,
        ).to(self._qwen_device)
        
        outputs = self.qwen_model(**inputs, output_hidden_states=True)
        
        action_token_ids = [
            self.qwen_tokenizer.encode(str(a), add_special_tokens=False)[0]
            for a in range(4)
        ]
        
        seq_lens = inputs["attention_mask"].sum(dim=1) - 1  
        
        all_logits = []
        all_hidden = []
        
        for i in range(batch_size):
            last_pos = seq_lens[i]
            last_token_logits = outputs.logits[i, last_pos, :]
            action_logits = last_token_logits[action_token_ids]
            all_logits.append(action_logits)
            
            last_hidden = outputs.hidden_states[-1][i, last_pos, :]
            all_hidden.append(last_hidden)
        
        return torch.stack(all_logits), torch.stack(all_hidden)

    def forward(self, obs, deterministic: bool = False):
        action_logits, hidden = self._get_qwen_features(obs)

        values = self.value_net(hidden)

        distribution = CategoricalDistribution(self.action_space.n)
        distribution.proba_distribution(action_logits)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        action_logits, hidden = self._get_qwen_features(obs)

        values = self.value_net(hidden)

        distribution = CategoricalDistribution(self.action_space.n)
        distribution.proba_distribution(action_logits)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def predict_values(self, obs):
        _, hidden = self._get_qwen_features(obs)
        return self.value_net(hidden)

    def _predict(self, observation, deterministic: bool = False):
        actions, _, _ = self.forward(observation, deterministic)
        return actions