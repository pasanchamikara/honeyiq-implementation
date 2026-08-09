# Appendix A — API Reference

This appendix summarises the public interfaces of each module, including the extensions introduced in this version (§3.2.4, §3.9, §3.10). Full docstrings and implementation details are available in the source code under the corresponding `defender/`, `attacker/`, `environment/`, `evaluation/`, and `opencanary_integration/` directories.

## A.1 Attacker

### `attacker.attack_types`

Attack type and kill chain enumerations, plus the traffic-realism constants introduced in §3.2.4:

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
ATTACK_PRIMARY_STAGE: Dict[int, int] # AttackType -> KillChainStage

# New in this version (§3.2.4)
INTENSITY_LOGNORMAL_SIGMA: float                        # 0.35
INTENSITY_SCALED_FEATURES: FrozenSet[str]                # 8 volume-shaped feature names
NORMAL_PERSONA_DISTRIBUTIONS: Dict[str, Dict[str, Tuple]] # 3 benign personas
NORMAL_PERSONA_WEIGHTS: Dict[str, float]                  # {"casual_user": 0.70, "crawler": 0.20, "monitoring_probe": 0.10}
```

### `attacker.transition_model`

Markov transition model interface (unchanged):

```python
class TransitionModel:
    def __init__(self, intent: AttackerIntent = OPPORTUNISTIC,
                 seed: int | None = None) -> None
    def next_attack(self, current: AttackType) -> AttackType
    def next_stage(self, current: KillChainStage) -> KillChainStage
    def get_attack_probabilities(self, current: AttackType
                                ) -> np.ndarray     # shape (10,)
    def get_stage_probabilities(self, current: KillChainStage
                               ) -> np.ndarray      # shape (7,)
    def get_attack_matrix(self) -> np.ndarray       # shape (10, 10)
    def get_stage_matrix(self) -> np.ndarray        # shape (7, 7)
```

### `attacker.attacker`

`Attacker` interface, extended with the session-profile mechanism of §3.2.4:

```python
class Attacker:
    current_attack: AttackType
    current_stage:  KillChainStage
    attack_count:   int
    step_count:     int

    def __init__(self, intent: AttackerIntent = OPPORTUNISTIC,
                 seed: int | None = None) -> None

    def reset(self) -> None
    # Re-seeds RNG (if seeded) and draws a new session profile

    def _draw_session_profile(self) -> None
    # Sets self._intensity (float) and self._benign_persona (str);
    # called from __init__ and reset()

    def step(self) -> Dict[str, Any]
    # {attack_type, kill_chain_stage, intent, attack_count, step_count,
    #  features, is_attack, next_probabilities, stage_probabilities}

    def get_state_info(self) -> Dict[str, Any]

    def _simulate_features(
        self, attack_type: AttackType, *,
        intensity: float | None = None,   # None = use self._intensity
        persona:   str | None = None,     # None = use self._benign_persona; NORMAL only
    ) -> Dict[str, float]                 # 15 keys
```

## A.2 Defender

### `defender.honeypot`

Honeypot action and reward computation (unchanged):

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

Stage-Escalation Decision Matrix interface, extended with the R4 override and the optional `AdaptiveThresholds` attachment (§3.4, §3.10):

```python
ESC_LOW_THRESHOLD    = 0.35
ESC_HIGH_THRESHOLD   = 0.65
RATE_THRESHOLD       = 0.80
REPUTATION_THRESHOLD = 0.60   # new in this version

class MatrixPolicy:
    def __init__(
        self,
        default_intent: AttackerIntent = OPPORTUNISTIC,
        adaptive_thresholds: Optional["AdaptiveThresholds"] = None,  # new
    ) -> None

    def decide_from_state(
        self, state: np.ndarray,        # shape (24,)
        reputation: float = 0.0,        # new
    ) -> Tuple[HoneypotAction, dict]

    def decide(
        self,
        current_stage:   KillChainStage,
        current_attack:  AttackType,
        escalation_rate: float,
        intent: Optional[AttackerIntent] = None,
        reputation: float = 0.0,        # new
    ) -> Tuple[HoneypotAction, dict]
    # info dict keys:
    #   stage, attack_type, intent, escalation_risk,
    #   escalation_band, base_action, reputation, override_applied,
    #   final_action, composite_risk

    @staticmethod
    def get_matrix() -> List[List[str]]          # 7x3 action names
    @staticmethod
    def get_matrix_actions() -> List[List[HoneypotAction]]  # 7x3 raw enum values
    @staticmethod
    def get_full_matrix_for_intent(
        intent: AttackerIntent
    ) -> np.ndarray  # shape (7,) -- one action per stage
```

### `defender.adaptive_thresholds` (new in this version, §3.10.2)

```python
@dataclass
class AdaptiveThresholds:
    initial_threshold: float = 0.80
    target_rate:        float = 0.10
    tolerance:           float = 0.03
    step:                 float = 0.01
    bound:                float = 0.10
    observation_window:  int   = 200

    def record(self, r3_condition: bool) -> None
    @property
    def threshold(self) -> float
```

### `defender.dqn`

DQN agent interface (unchanged; retained for comparison, §3.5, §5.2–§5.3):

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

Random Forest attack classifier interface (unchanged; training-data generation now benefits from the §3.2.4 independent-sample override):

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

### `defender.defender`

Orchestrator interface, extended with the `reputation` pass-through:

```python
class Defender:
    epsilon:    float   # stub property, DQN-era compatibility
    steps_done: int      # stub property

    def __init__(self, classifier_config: dict | None = None,
                 train_classifier: bool = True, seed: int = 42,
                 default_intent: AttackerIntent = OPPORTUNISTIC,
                 dqn_config: dict | None = None) -> None  # accepted, ignored

    def initialize_classifier(self, n_samples_per_class: int = 600) -> None

    def observe(self, state: np.ndarray, features: dict,
                training: bool = True,
                reputation: float = 0.0) -> Tuple[int, AttackType]  # new param

    def get_attack_probabilities(self, features: dict) -> np.ndarray
    def get_decision_info(self, state: np.ndarray, features: dict) -> dict
    def learn(self, state, action, reward, next_state, done) -> Optional[float]
    def save(self, model_dir: str = "models/") -> None
    def load(self, model_dir: str = "models/") -> None
    def policy_matrix(self) -> List[List[str]]
```

## A.3 Environment

### `environment.cyber_env`

Gymnasium environment interface, extended with the escalation-mode selection of §3.9:

```python
STATE_DIM  = 24
ACTION_DIM =  5

def encode_state(attack_type, kill_chain_stage, threat_level,
                  attack_count, escalation_rate, intent) -> np.ndarray  # (24,)
# Shared by CyberSecurityEnv and opencanary_integration's state_builder

class CyberSecurityEnv(gymnasium.Env):
    observation_space: Box(shape=(24,), dtype=float32)
    action_space:      Discrete(5)
    current_threat: float              # property
    current_state:  Optional[np.ndarray]  # property

    def __init__(
        self,
        attacker_intent: AttackerIntent = OPPORTUNISTIC,
        max_steps: int = 500,
        escalation_window: int = 20,
        escalation_mode: str = "window",       # new: "window" | "ema"
        escalation_ema_alpha: float = 0.15,    # new
        seed: int | None = None,
        render_mode: str | None = None,
        benign_ratio: float = 0.0,
    ) -> None
    # Raises ValueError if escalation_mode not in ("window", "ema")

    def reset(
        self, seed=None, options=None
    ) -> Tuple[np.ndarray, dict]
    # info contains: features, attack_type, kill_chain_stage, threat_level,
    #                is_attack, escalation_rate, escalation_window_rate,
    #                escalation_ema (last two are new)

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]
    # Returns: (next_state, reward, terminated, truncated, info)
```

## A.4 Evaluation

### `evaluation.metrics`

`MetricsCollector` interface (unchanged):

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
```

## A.5 OpenCanary Integration (new in this version, §3.10)

### `opencanary_integration.ingest`

```python
class OpenCanaryEvent(BaseModel):        # pydantic model
    dst_host: str; dst_port: int; logdata: dict; logtype: int
    node_id: str; src_host: str; src_port: int
    utc_time: str; local_time: str
    @property
    def service_name(self) -> str

def map_logtype(event: OpenCanaryEvent) -> AttackType
def initial_stage_for(attack_type: AttackType) -> KillChainStage
```

### `opencanary_integration.engine`

```python
@dataclass
class SessionState:
    src_ip: str
    current_attack: AttackType; current_stage: KillChainStage
    attack_count: int; event_count: int
    recent_attacks: deque; escalation_ema: float
    last_seen: datetime; inferred_intent: AttackerIntent
    reputation: float
    @property
    def escalation_rate(self) -> float

class SessionTracker:
    reputation: "ReputationTracker"   # public
    def __init__(self, ttl_seconds: int = 3600, escalation_window: int = 20,
                 escalation_ema_alpha: float = 0.15,
                 sweep_interval_seconds: int = 60) -> None
    def update(self, src_ip: str, attack_type: AttackType) -> SessionState
    def get(self, src_ip: str) -> Optional[SessionState]
    def remove(self, src_ip: str) -> None
    def all_sessions(self) -> Dict[str, SessionState]

class ReputationTracker:
    def __init__(self, decay_half_life_seconds: float = 6*3600,
                 offense_increment: float = 0.25, max_score: float = 1.0,
                 stale_after_seconds: float = 30*24*3600,
                 sweep_interval_seconds: int = 300) -> None
    def record_offense(self, src_ip: str, severity: float) -> float
    def get(self, src_ip: str) -> float
    def reset(self, src_ip: str) -> None

def build_state(session: SessionState, escalation_mode: str = "window") -> np.ndarray

class EscalationPredictor:
    def __init__(self, intent: AttackerIntent = OPPORTUNISTIC) -> None
    def next_attack_probs(self, current: AttackType) -> np.ndarray
    def next_stage_probs(self, current: KillChainStage) -> np.ndarray
    def escalation_risk(self, current_stage: KillChainStage,
                         probs: Optional[np.ndarray] = None) -> float
    def most_likely_next_stage(self, current: KillChainStage) -> KillChainStage
    def most_likely_next_attack(self, current: AttackType) -> AttackType

class PolicyEngine:
    def __init__(self, model_dir: str = "models/",
                 default_intent: str = "OPPORTUNISTIC") -> None
    def decide(self, state: np.ndarray, features: Optional[dict] = None,
               reputation: float = 0.0) -> Tuple[HoneypotAction, Optional[AttackType]]
    def decision_info(self, state: np.ndarray) -> dict
    def reload(self, model_dir: str = "models/") -> None
    @property
    def classifier_fitted(self) -> bool
```

### `opencanary_integration.emulator`

```python
class OpenCanaryEventGenerator:
    def __init__(self, node_id: str = "honeypot-emulator-01",
                 dst_host: str = "192.168.1.100", seed: int | None = None) -> None
    def generate(self, scenario: str, src_ip: Optional[str] = None) -> OpenCanaryEvent
    def generate_sequence(self, scenarios: List[str], src_ip: Optional[str] = None,
                           delay_ms: float = 0.0) -> List[OpenCanaryEvent]
    def generate_kill_chain(self, src_ip: Optional[str] = None) -> List[OpenCanaryEvent]
    def available_scenarios(self) -> List[str]

class DummyHoneypot:
    def __init__(self, audit_file: Optional[str] = None, verbose: bool = True) -> None
    def apply_action_sync(self, src_ip, action, attack_type="UNKNOWN",
                           stage="UNKNOWN", threat_level=0.0, event_id="") -> None
    def get_ip_status(self, src_ip: str) -> dict
    def get_all_sessions(self) -> List[dict]
    def get_action_log(self) -> List[dict]
    def close(self) -> None

class EmulatorScenario:
    def __init__(self, model_dir: str = "models/", intent: str = "OPPORTUNISTIC",
                 audit_file: Optional[str] = None, verbose: bool = True,
                 escalation_mode: str = "window") -> None
    def run_event(self, event: OpenCanaryEvent) -> dict
    # {event_id, src_ip, logtype, service, attack_type, stage, threat_level,
    #  escalation_risk, reputation, action, stage_probs, attack_probs}
```
