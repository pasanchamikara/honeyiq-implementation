# Appendix A — API Reference

This appendix summarises the public interfaces of each module. Full docstrings and implementation details are available in the source code under the corresponding `defender/`, `attacker/`, `environment/`, and `evaluation/` directories.

## A.1 Attacker

### `attacker.attack_types`

Attack type and kill chain enumerations:

```python
class AttackType(IntEnum):
    NORMAL=0, RECONNAISSANCE=1, ANALYSIS=2, FUZZERS=3,
    GENERIC=4, EXPLOITS=5, SHELLCODE=6, BACKDOORS=7,
    DOS=8, WORMS=9

    @classmethod def names() -> List[str]
    @classmethod def count() -> int

class KillChainStage(IntEnum):
    RECONNAISSANCE=0, WEAPONIZATION=1, DELIVERY=2,
    EXPLOITATION=3, INSTALLATION=4,
    COMMAND_AND_CTRL=5, ACTIONS_ON_OBJ=6

class AttackerIntent(IntEnum):
    STEALTHY=0, AGGRESSIVE=1, TARGETED=2, OPPORTUNISTIC=3

# Severity and kill-chain weight dicts
ATTACK_SEVERITY: Dict[int, float]    # 0.00 .. 0.90
KILL_CHAIN_WEIGHT: Dict[int, float]  # 0.10 .. 1.00
```

### `attacker.transition_model`

Markov transition model interface:

```python
class TransitionModel:
    def __init__(self, intent: AttackerIntent = OPPORTUNISTIC) -> None
    def next_attack(self, current: AttackType) -> AttackType
    def next_stage(self, current: KillChainStage) -> KillChainStage
    def get_attack_probabilities(self, current: AttackType
                                ) -> np.ndarray     # shape (10,)
    def get_stage_probabilities(self, current: KillChainStage
                               ) -> np.ndarray      # shape (7,)
```

### `attacker.attacker`

`AttackerAgent` interface:

```python
class AttackerAgent:
    def __init__(self, intent: AttackerIntent = OPPORTUNISTIC,
                 seed: int = 42) -> None
    def reset(self) -> None
    def step(self) -> Tuple[AttackType, KillChainStage,
                            np.ndarray]  # (type, stage, 15 features)
```

## A.2 Defender

### `defender.honeypot`

Honeypot action and reward computation:

```python
class HoneypotAction(IntEnum):
    ALLOW=0, LOG=1, TROLL=2, BLOCK=3, ALERT=4

def compute_threat_level(
    attack_type: AttackType,
    kill_chain_stage: KillChainStage,
    escalation_rate: float,
    attack_count: int
) -> float  # in [0.0, 1.0]

def threat_band(threat_level: float) -> str
# Returns: "benign" | "low" | "medium" | "high" | "critical"

def compute_reward(
    action: HoneypotAction,
    threat_level: float,
    attack_type: AttackType,
    kill_chain_stage: KillChainStage,
    is_attack: bool
) -> float
```

### `defender.matrix_policy` (SEDM)

Stage-Escalation Decision Matrix interface:

```python
class MatrixPolicy:
    def __init__(
        self,
        default_intent: AttackerIntent = OPPORTUNISTIC
    ) -> None

    def decide_from_state(
        self, state: np.ndarray  # shape (24,)
    ) -> Tuple[HoneypotAction, dict]

    def decide(
        self,
        current_stage:   KillChainStage,
        current_attack:  AttackType,
        escalation_rate: float,
        intent: Optional[AttackerIntent] = None
    ) -> Tuple[HoneypotAction, dict]
    # info dict keys:
    #   stage, attack_type, intent, escalation_risk,
    #   escalation_band, base_action, override_applied,
    #   final_action, composite_risk

    @staticmethod
    def get_matrix() -> List[List[str]]
    # Returns 7x3 list of action names

    @staticmethod
    def get_full_matrix_for_intent(
        intent: AttackerIntent
    ) -> np.ndarray  # shape (7,) -- one action per stage
```

### `defender.dqn`

DQN agent interface:

```python
class DQNAgent:
    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.997,
        buffer_capacity: int = 15000,
        batch_size: int = 64,
        target_update_freq: int = 150
    ) -> None

    def select_action(self, state: np.ndarray,
                      training: bool = True) -> int
    def store_transition(self, s, a, r, s_next, done) -> None
    def update(self) -> Optional[float]   # loss or None
    def save(self, path: str) -> None
    def load(self, path: str) -> None

    # Properties
    epsilon: float
    steps_done: int

    def q_values(self, state: np.ndarray) -> np.ndarray  # shape (5,)
```

### `defender.classifier`

Random Forest attack classifier interface:

```python
class AttackClassifier:
    def __init__(self, n_estimators: int = 150,
                 max_depth: int = 20,
                 n_jobs: int = 1) -> None

    def generate_training_data(
        self,
        n_samples_per_class: int = 600,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray]

    def fit_from_simulation(
        self, n_samples_per_class: int = 600, seed: int = 42
    ) -> None

    def predict(self, features: dict) -> AttackType
    def predict_proba(self, features: dict) -> np.ndarray  # (10,)
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray

    def evaluate(self, n_test_per_class: int = 200,
                 seed: int = 999) -> dict
    # Returns: {"accuracy": float, "report": dict}

    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

## A.3 Environment

### `environment.CyberSecurityEnv`

Gymnasium environment interface:

```python
class CyberSecurityEnv(gymnasium.Env):
    observation_space: Box(shape=(24,), dtype=float32)
    action_space:      Discrete(5)

    def __init__(
        self,
        intent: AttackerIntent = OPPORTUNISTIC,
        max_steps: int = 500,
        seed: int = 42
    ) -> None

    def reset(
        self, seed=None, options=None
    ) -> Tuple[np.ndarray, dict]
    # info contains: features, attack_type, kill_chain_stage,
    #                threat_level, is_attack

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]
    # Returns: (next_state, reward, terminated, truncated, info)
```

## A.4 Evaluation

### `evaluation.metrics`

`MetricsCollector` interface:

```python
@dataclass
class StepRecord:
    episode: int; step: int; action: int; reward: float
    attack_type: int; kill_chain_stage: int
    threat_level: float; is_attack: bool
    predicted_attack: int; loss: Optional[float]
    escalation_rate: float

@dataclass
class EpisodeRecord:
    episode: int; total_reward: float; steps: int
    detection_rate: float; false_positive_rate: float
    avg_threat_level: float; avg_loss: float
    kill_chain_dist: dict; action_dist: dict

class MetricsCollector:
    def __init__(self, log_dir: str = "logs/") -> None

    def record_step(self, episode, step, action, reward,
                    info, predicted_attack, loss) -> None
    def end_episode(self, episode: int) -> EpisodeRecord
    def summary_report(self) -> dict
    def save_csv(self) -> None
    def plot_training_curves(self) -> None
    def plot_kill_chain_heatmap(self) -> None
    def plot_attack_progression(
        self, step_records: List[StepRecord]
    ) -> None
    def analyze(self) -> None
```
