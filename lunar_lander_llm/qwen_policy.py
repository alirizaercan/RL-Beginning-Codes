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

        # ✅ nn.Module.__init__() çağrılmadan önce attribute set etmek için
        # object.__setattr__ kullanılmalı — aksi hâlde PyTorch hata verir.
        object.__setattr__(self, "qwen_model", None)
        object.__setattr__(self, "qwen_tokenizer", None)

        # Qwen'i super().__init__() ÖNCE yükle.
        # super().__init__() içinde _build_mlp_extractor() çağrılır ve
        # orada self.qwen_model.config.hidden_size'a ihtiyaç duyulur.
        self._load_qwen_model()

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Model yükleme
    # ------------------------------------------------------------------

    def _load_qwen_model(self):
        print(f"Loading Qwen model from {self.qwen_model_path}...")

        tokenizer = AutoTokenizer.from_pretrained(
            self.qwen_model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ✅ float32 — float16'da gradient hesaplanamaz
        qwen = AutoModelForCausalLM.from_pretrained(
            self.qwen_model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        if hasattr(qwen, "gradient_checkpointing_enable"):
            qwen.gradient_checkpointing_enable()

        for param in qwen.parameters():
            param.requires_grad = True
        qwen.train()

        # ✅ nn.Module.__init__() çağrılmadan önce attribute set etmenin
        # tek güvenli yolu object.__setattr__
        object.__setattr__(self, "qwen_model", qwen)
        object.__setattr__(self, "qwen_tokenizer", tokenizer)

        print("Qwen model loaded and ready for fine-tuning!")
        print(
            f"Trainable parameters: "
            f"{sum(p.numel() for p in qwen.parameters() if p.requires_grad):,}"
        )

    # ------------------------------------------------------------------
    # MLP extractor — SB3'ün beklediği arayüzü karşılar
    # ------------------------------------------------------------------

    def _build_mlp_extractor(self):
        """
        SB3, action_net ve value_net'i latent_dim değerlerine göre inşa eder.
        Action logitler Qwen'den geldiği için action_net'i devre dışı bırakıyoruz;
        value_net'i de kendimiz tanımlıyoruz.
        Stub extractor yalnızca latent_dim attribute'larını taşır.
        """
        qwen_hidden_size = self.qwen_model.config.hidden_size

        class _StubExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_dim_pi = 4                # 4 action logit
                self.latent_dim_vf = qwen_hidden_size # Qwen hidden size

            def forward(self, features):
                return features, features

            def forward_actor(self, features):
                return features

            def forward_critic(self, features):
                return features

        self.mlp_extractor = _StubExtractor()

        # ✅ Critic: Qwen'in son hidden state'ini alır
        self.value_net = nn.Sequential(
            nn.Linear(qwen_hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # SB3'ün oluşturduğu varsayılan action_net'i etkisiz hâle getir
        self.action_net = nn.Identity()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def obs_to_prompt(self, obs: torch.Tensor) -> str:
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

    # ------------------------------------------------------------------
    # Qwen forward — action logits + critic hidden state
    # ------------------------------------------------------------------

    def _get_qwen_features(self, obs: torch.Tensor):
        """
        Her gözlem için Qwen'i çalıştırır.

        Döndürür:
            action_logits : Tensor[batch, 4]  — gradient korunur
            hidden_states : Tensor[batch, H]  — critic için
        """
        batch_size = obs.shape[0]
        all_logits = []
        all_hidden = []

        # Token id'leri bir kez hesapla
        action_token_ids = [
            self.qwen_tokenizer.encode(str(a), add_special_tokens=False)[0]
            for a in range(4)
        ]

        for i in range(batch_size):
            prompt = self.obs_to_prompt(obs[i : i + 1])

            inputs = self.qwen_tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(obs.device)

            outputs = self.qwen_model(**inputs, output_hidden_states=True)

            # Action logits — gradient burada korunur
            last_token_logits = outputs.logits[0, -1, :]       # [vocab]
            action_logits = last_token_logits[action_token_ids] # [4]
            all_logits.append(action_logits)

            # Critic için son hidden state
            last_hidden = outputs.hidden_states[-1][0, -1, :]  # [H]
            all_hidden.append(last_hidden)

        return torch.stack(all_logits), torch.stack(all_hidden) # [B,4], [B,H]

    # ------------------------------------------------------------------
    # SB3 API
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        action_logits, hidden = self._get_qwen_features(obs)

        values = self.value_net(hidden)

        distribution = CategoricalDistribution(self.action_space.n)
        distribution.proba_distribution(action_logits)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        action_logits, hidden = self._get_qwen_features(obs)

        values = self.value_net(hidden)

        distribution = CategoricalDistribution(self.action_space.n)
        distribution.proba_distribution(action_logits)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        _, hidden = self._get_qwen_features(obs)
        return self.value_net(hidden)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        actions, _, _ = self.forward(observation, deterministic)
        return actions