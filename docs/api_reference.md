# API Reference

## `attacker.attack_types`

### Enumerations

```python
class AttackType(IntEnum):
    NORMAL=0, RECONNAISSANCE=1, ANALYSIS=2, FUZZERS=3, EXPLOITS=4,
    BACKDOORS=5, SHELLCODE=6, GENERIC=7, DOS=8, WORMS=9

    @classmethod def names() -> List[str]
    @classmethod def count() -> int

class KillChainStage(IntEnum):
    RECONNAISSANCE=0, WEAPONIZATION=1, DELIVERY=2, EXPLOITATION=3,
    INSTALLATION=4, COMMAND_AND_CTRL=5, ACTIONS_ON_OBJ=6

    @classmethod def names() -> List[str]
    @classmethod def count() -> int

class AttackerIntent(IntEnum):
    STEALTHY=0, AGGRESSIVE=1, TARGETED=2, OPPORTUNISTIC=3

    @classmethod def names() -> List[str]
    @classmethod def count() -> int
```

### Constants

```python
ATTACK_SEVERITY:      Dict[int, float]  # AttackType → [0.0, 0.90]
KILL_CHAIN_WEIGHT:    Dict[int, float]  # KillChainStage → [0.10, 1.00]
ATTACK_PRIMARY_STAGE: Dict[int, int]    # AttackType → KillChainStage
FEATURE_DISTRIBUTIONS: Dict[int, Dict[str, Tuple]]  # per-attack feature specs
FEATURE_NAMES:        List[str]         # ordered list of 15 feature names
```

---

## `attacker.transition_model`

```python
class TransitionModel:
    def __init__(self,
                 intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
                 seed: int | None = None) -> None

    def next_attack(self, current: AttackType) -> AttackType
    def next_stage(self, current: KillChainStage) -> KillChainStage

    def get_attack_probabilities(self, current: AttackType) -> np.ndarray  # (10,)
    def get_stage_probabilities(self, current: KillChainStage) -> np.ndarray  # (7,)
    def get_attack_matrix(self) -> np.ndarray  # (10, 10)
    def get_stage_matrix(self) -> np.ndarray   # (7, 7)
```

---

## `attacker.attacker`

```python
class Attacker:
    current_attack: AttackType
    current_stage:  KillChainStage
    attack_count:   int
    step_count:     int

    def __init__(self,
                 intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
                 seed: int | None = None) -> None

    def reset(self) -> None

    def step(self) -> Dict[str, Any]:
        # Returns:
        # {
        #   "attack_type":        AttackType,
        #   "kill_chain_stage":   KillChainStage,
        #   "intent":             AttackerIntent,
        #   "attack_count":       int,
        #   "step_count":         int,
        #   "features":           Dict[str, float],
        #   "is_attack":          bool,
        #   "next_probabilities": np.ndarray,  # (10,)
        #   "stage_probabilities":np.ndarray,  # (7,)
        # }

    def get_state_info(self) -> Dict[str, Any]
```

---

## `defender.honeypot`

```python
class HoneypotAction(IntEnum):
    ALLOW=0, LOG=1, TROLL=2, BLOCK=3, ALERT=4

    @classmethod def names() -> List[str]
    @classmethod def count() -> int

def threat_band(threat_level: float) -> str
    # Returns: "benign" | "low" | "medium" | "high" | "critical"

def compute_threat_level(
    attack_type:      AttackType,
    kill_chain_stage: KillChainStage,
    escalation_rate:  float,
    attack_count:     int,
) -> float  # [0.0, 1.0]

def compute_reward(
    action:           int,
    threat_level:     float,
    is_attack:        bool,
    kill_chain_stage: KillChainStage,
    attack_type:      AttackType,
) -> float
```

---

## `defender.dqn`

```python
class DQNNetwork(nn.Module):
    def __init__(self,
                 state_dim:   int = 24,
                 action_dim:  int = 5,
                 hidden_dims: list[int] | None = None) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor  # → (batch, action_dim)

class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None
    def push(self, state, action, reward, next_state, done) -> None
    def sample(self, batch_size: int, device: torch.device) -> tuple[5 × Tensor]
    def __len__(self) -> int

class DQNAgent:
    epsilon:     float
    steps_done:  int
    device:      torch.device
    policy_net:  DQNNetwork
    target_net:  DQNNetwork

    def __init__(self,
                 state_dim:          int   = 24,
                 action_dim:         int   = 5,
                 hidden_dims:        list[int] | None = None,
                 lr:                 float = 1e-3,
                 gamma:              float = 0.99,
                 epsilon_start:      float = 1.0,
                 epsilon_end:        float = 0.05,
                 epsilon_decay:      float = 0.995,
                 batch_size:         int   = 64,
                 target_update_freq: int   = 100,
                 buffer_capacity:    int   = 10_000,
                 device:             str   = "auto") -> None

    def select_action(self, state: np.ndarray, training: bool = True) -> int
    def store_transition(self, state, action, reward, next_state, done) -> None
    def update(self) -> float | None   # returns loss, or None if buffer not warm
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

---

## `defender.classifier`

```python
class AttackClassifier:
    is_fitted:     bool
    feature_names: List[str]

    def __init__(self,
                 n_estimators: int = 150,
                 max_depth:    int | None = 20,
                 random_state: int = 42) -> None

    def generate_training_data(self,
                                n_samples_per_class: int = 600,
                                seed: int = 42) -> tuple[pd.DataFrame, pd.Series]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AttackClassifier"
    def fit_from_simulation(self,
                             n_samples_per_class: int = 600,
                             seed: int = 42) -> "AttackClassifier"

    def predict(self, features: dict[str, float]) -> AttackType
    def predict_proba(self, features: dict[str, float]) -> np.ndarray  # (10,)
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray             # (N,) int

    def evaluate(self,
                 n_test_per_class: int = 200,
                 seed: int = 99) -> dict
        # Returns: {"accuracy": float, "report": dict}

    def save(self, path: str) -> None
    def load(self, path: str) -> "AttackClassifier"
```

---

## `defender.defender`

```python
class Defender:
    dqn_agent:  DQNAgent
    classifier: AttackClassifier
    epsilon:    float       # property → dqn_agent.epsilon
    steps_done: int         # property → dqn_agent.steps_done

    def __init__(self,
                 dqn_config:        dict | None = None,
                 classifier_config: dict | None = None,
                 train_classifier:  bool = True,
                 seed:              int  = 42) -> None

    def initialize_classifier(self, n_samples_per_class: int = 600) -> None

    def observe(self,
                state:    np.ndarray,
                features: dict[str, float],
                training: bool = True) -> tuple[int, AttackType]
        # Returns: (action, predicted_attack)

    def get_attack_probabilities(self, features: dict[str, float]) -> np.ndarray  # (10,)

    def learn(self,
              state:      np.ndarray,
              action:     int,
              reward:     float,
              next_state: np.ndarray,
              done:       bool) -> float | None

    def save(self, model_dir: str = "models/") -> None
    def load(self, model_dir: str = "models/") -> None
    def q_values(self, state: np.ndarray) -> np.ndarray  # (5,)
```

---

## `environment.cyber_env`

```python
STATE_DIM  = 24
ACTION_DIM =  5

class CyberSecurityEnv(gym.Env):
    current_threat: float    # property
    current_state:  np.ndarray | None  # property

    def __init__(self,
                 attacker_intent:   AttackerIntent = AttackerIntent.OPPORTUNISTIC,
                 max_steps:         int = 500,
                 escalation_window: int = 20,
                 seed:              int | None = None,
                 render_mode:       str | None = None) -> None

    def reset(self,
              seed:    int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]

    def render(self) -> str | None

    def close(self) -> None
```

---

## `evaluation.metrics`

```python
@dataclass
class StepRecord:
    episode: int; step: int; action: int; reward: float
    attack_type: int; kill_chain_stage: int; threat_level: float
    is_attack: bool; predicted_attack: int; loss: float | None
    escalation_rate: float

@dataclass
class EpisodeRecord:
    episode: int; total_reward: float; steps: int
    detection_rate: float; false_positive_rate: float
    avg_threat_level: float; avg_loss: float
    kill_chain_dist: Dict[str, int]; action_dist: Dict[str, int]

class MetricsCollector:
    episodes: List[EpisodeRecord]

    def __init__(self, log_dir: str = "logs/") -> None

    def record_step(self,
                    episode: int, step: int, action: int,
                    reward: float, info: dict,
                    predicted_attack: AttackType,
                    loss: float | None) -> None

    def end_episode(self, episode: int) -> EpisodeRecord

    def summary_report(self) -> dict
        # Keys: total_episodes, mean_reward, std_reward, best_episode_reward,
        #       mean_detection_rate, mean_false_positive_rate, mean_threat_level

    def save_csv(self, path: str | None = None) -> None

    def plot_training_curves(self,
                              save_path: str | None = None,
                              rolling_window: int = 10) -> None

    def plot_kill_chain_heatmap(self, save_path: str | None = None) -> None

    def plot_attack_progression(self,
                                 step_records: List[StepRecord],
                                 save_path: str | None = None) -> None
```
