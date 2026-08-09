# API Reference

Current as of this implementation round. Signatures shown are the actual
current source, not the pre-SEDM/pre-dynamic-response versions documented
in `docs/api_reference.md`.

## `attacker.attack_types`

```python
class AttackType(IntEnum):
    NORMAL=0, RECONNAISSANCE=1, ANALYSIS=2, FUZZERS=3, EXPLOITS=4,
    BACKDOORS=5, SHELLCODE=6, GENERIC=7, DOS=8, WORMS=9
    @classmethod def names() -> List[str]
    @classmethod def count() -> int

class KillChainStage(IntEnum):
    RECONNAISSANCE=0, WEAPONIZATION=1, DELIVERY=2, EXPLOITATION=3,
    INSTALLATION=4, COMMAND_AND_CTRL=5, ACTIONS_ON_OBJ=6
    # .names(), .count() as above

class AttackerIntent(IntEnum):
    STEALTHY=0, AGGRESSIVE=1, TARGETED=2, OPPORTUNISTIC=3
    # .names(), .count() as above

# Constants
ATTACK_SEVERITY:      Dict[int, float]              # AttackType → [0.0, 0.90]
KILL_CHAIN_WEIGHT:    Dict[int, float]               # KillChainStage → [0.10, 1.00]
ATTACK_PRIMARY_STAGE: Dict[int, int]                 # AttackType → KillChainStage
FEATURE_DISTRIBUTIONS: Dict[int, Dict[str, Tuple]]   # per-attack-type feature specs
FEATURE_NAMES:        List[str]                      # 15 ordered feature names

# New this round
INTENSITY_LOGNORMAL_SIGMA:   float                          # 0.35
INTENSITY_SCALED_FEATURES:   frozenset[str]                 # 8 volume-shaped feature names
NORMAL_PERSONA_DISTRIBUTIONS: Dict[str, Dict[str, Tuple]]    # 3 benign personas
NORMAL_PERSONA_WEIGHTS:       Dict[str, float]               # persona sampling weights
```

## `attacker.transition_model`

```python
class TransitionModel:
    def __init__(self, intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
                 seed: int | None = None) -> None

    def next_attack(self, current: AttackType) -> AttackType
    def next_stage(self, current: KillChainStage) -> KillChainStage
    def get_attack_probabilities(self, current: AttackType) -> np.ndarray   # (10,)
    def get_stage_probabilities(self, current: KillChainStage) -> np.ndarray  # (7,)
    def get_attack_matrix(self) -> np.ndarray   # (10, 10)
    def get_stage_matrix(self) -> np.ndarray    # (7, 7)
```

## `attacker.attacker`

```python
class Attacker:
    current_attack: AttackType
    current_stage:  KillChainStage
    attack_count:   int
    step_count:     int

    def __init__(self, intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
                 seed: int | None = None) -> None

    def reset(self) -> None
        # Re-seeds RNG (if seeded) and draws a new session profile

    def _draw_session_profile(self) -> None
        # Sets self._intensity (float) and self._benign_persona (str)

    def step(self) -> Dict[str, Any]
        # {attack_type, kill_chain_stage, intent, attack_count, step_count,
        #  features, is_attack, next_probabilities, stage_probabilities}

    def get_state_info(self) -> Dict[str, Any]

    def _simulate_features(
        self, attack_type: AttackType, *,
        intensity: float | None = None,   # None = use self._intensity
        persona:   str | None = None,     # None = use self._benign_persona; NORMAL only
    ) -> Dict[str, float]   # 15 keys, matches FEATURE_NAMES
```

## `defender.honeypot`

```python
class HoneypotAction(IntEnum):
    ALLOW=0, LOG=1, TROLL=2, BLOCK=3, ALERT=4
    # .names(), .count()

def threat_band(threat_level: float) -> str
    # "benign" | "low" | "medium" | "high" | "critical"

def compute_threat_level(attack_type, kill_chain_stage,
                          escalation_rate, attack_count) -> float   # [0, 1]

def compute_reward(action, threat_level, is_attack,
                    kill_chain_stage, attack_type) -> float
```

## `defender.classifier`

```python
class AttackClassifier:
    is_fitted:     bool
    feature_names: List[str]

    def __init__(self, n_estimators: int = 150, max_depth: int | None = 20,
                 random_state: int = 42, n_jobs: int | None = 1) -> None

    def generate_training_data(self, n_samples_per_class: int = 600,
                                seed: int = 42) -> tuple[pd.DataFrame, pd.Series]
        # Draws a fresh intensity/persona per sample (see attacker.md)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AttackClassifier"
    def fit_from_simulation(self, n_samples_per_class: int = 600,
                             seed: int = 42) -> "AttackClassifier"

    def predict(self, features: dict[str, float]) -> AttackType
    def predict_proba(self, features: dict[str, float]) -> np.ndarray  # (10,)
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray

    def evaluate(self, n_test_per_class: int = 200, seed: int = 99) -> dict
        # {"accuracy": float, "report": dict}

    def save(self, path: str) -> None
    def load(self, path: str) -> "AttackClassifier"
```

## `defender.matrix_policy`

```python
ESC_LOW_THRESHOLD    = 0.35
ESC_HIGH_THRESHOLD   = 0.65
RATE_THRESHOLD       = 0.80
REPUTATION_THRESHOLD = 0.60

class MatrixPolicy:
    def __init__(self, default_intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
                 adaptive_thresholds: AdaptiveThresholds | None = None) -> None

    def decide_from_state(self, state: np.ndarray,
                           reputation: float = 0.0) -> tuple[HoneypotAction, dict]

    def decide(self, current_stage: KillChainStage, current_attack: AttackType,
               escalation_rate: float, intent: AttackerIntent | None = None,
               reputation: float = 0.0) -> tuple[HoneypotAction, dict]
        # info: stage, attack_type, intent, escalation_risk, escalation_band,
        #       base_action, reputation, override_applied, final_action, composite_risk

    @staticmethod
    def get_matrix() -> list[list[str]]           # action names, 7x3
    @staticmethod
    def get_matrix_actions() -> list[list[HoneypotAction]]   # raw enum values, 7x3
    @staticmethod
    def get_full_matrix_for_intent(intent: AttackerIntent) -> np.ndarray  # (7,)
```

## `defender.adaptive_thresholds`

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

## `defender.defender`

```python
class Defender:
    classifier:    AttackClassifier
    matrix_policy: MatrixPolicy
    epsilon:       float   # stub property, always 0.0 (DQN-era compatibility)
    steps_done:    int     # stub property, always 0

    def __init__(self, classifier_config: dict | None = None,
                 train_classifier: bool = True, seed: int = 42,
                 default_intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
                 dqn_config: dict | None = None) -> None   # dqn_config accepted, ignored

    def initialize_classifier(self, n_samples_per_class: int = 600) -> None

    def observe(self, state: np.ndarray, features: dict[str, float],
                training: bool = True,   # ignored — SEDM is deterministic
                reputation: float = 0.0) -> tuple[int, AttackType]

    def get_attack_probabilities(self, features: dict[str, float]) -> np.ndarray

    def get_decision_info(self, state: np.ndarray, features: dict[str, float]) -> dict

    def learn(self, state, action, reward, next_state, done) -> float | None
        # No-op — always returns None. MatrixPolicy has no trainable parameters.

    def save(self, model_dir: str = "models/") -> None   # classifier only
    def load(self, model_dir: str = "models/") -> None

    def policy_matrix(self) -> list[list[str]]   # → MatrixPolicy.get_matrix()
```

## `environment.cyber_env`

```python
STATE_DIM  = 24
ACTION_DIM =  5
DEFAULT_EMA_ALPHA = 0.15

def encode_state(attack_type, kill_chain_stage, threat_level,
                  attack_count, escalation_rate, intent) -> np.ndarray   # (24,)
    # Shared by CyberSecurityEnv and opencanary_integration's state_builder

class CyberSecurityEnv(gym.Env):
    current_threat: float             # property
    current_state:  np.ndarray | None # property

    def __init__(self, attacker_intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC,
                 max_steps: int = 500, escalation_window: int = 20,
                 escalation_mode: str = "window",        # "window" | "ema"
                 escalation_ema_alpha: float = DEFAULT_EMA_ALPHA,
                 seed: int | None = None, render_mode: str | None = None,
                 benign_ratio: float = 0.0) -> None
        # Raises ValueError if escalation_mode not in ("window", "ema")

    def reset(self, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]
        # info always includes: step, attack_type, kill_chain_stage, threat_level,
        #   is_attack, features, escalation_rate, escalation_window_rate,
        #   escalation_ema, attack_count, action_name, next_probs, stage_probs

    def render(self) -> str | None
    def close(self) -> None
```

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
    def record_step(self, episode, step, action, reward, info,
                     predicted_attack, loss) -> None
    def end_episode(self, episode: int) -> EpisodeRecord
    def summary_report(self) -> dict
    def save_csv(self, path: str | None = None) -> None
    def plot_training_curves(self, save_path=None, rolling_window: int = 10) -> None
    def plot_kill_chain_heatmap(self, save_path: str | None = None) -> None
    def plot_attack_progression(self, step_records: List[StepRecord],
                                 save_path: str | None = None) -> None
```

## `opencanary_integration.ingest`

```python
# models.py
class OpenCanaryEvent(BaseModel):
    dst_host: str; dst_port: int; logdata: dict[str, Any]; logtype: int
    node_id: str; src_host: str; src_port: int
    utc_time: str; local_time: str
    @property
    def service_name(self) -> str

# logtype_map.py
def map_logtype(event: OpenCanaryEvent) -> AttackType
def initial_stage_for(attack_type: AttackType) -> KillChainStage
```

## `opencanary_integration.engine`

```python
# session_tracker.py
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
    reputation: ReputationTracker   # public
    def __init__(self, ttl_seconds: int = 3600, escalation_window: int = 20,
                 escalation_ema_alpha: float = 0.15,
                 sweep_interval_seconds: int = 60) -> None
    def update(self, src_ip: str, attack_type: AttackType) -> SessionState
    def get(self, src_ip: str) -> SessionState | None
    def remove(self, src_ip: str) -> None
    def all_sessions(self) -> dict[str, SessionState]

# reputation.py
class ReputationTracker:
    def __init__(self, decay_half_life_seconds: float = 6*3600,
                 offense_increment: float = 0.25, max_score: float = 1.0,
                 stale_after_seconds: float = 30*24*3600,
                 sweep_interval_seconds: int = 300) -> None
    def record_offense(self, src_ip: str, severity: float) -> float
    def get(self, src_ip: str) -> float
    def reset(self, src_ip: str) -> None

# state_builder.py
def build_state(session: SessionState, escalation_mode: str = "window") -> np.ndarray

# escalation_predictor.py
class EscalationPredictor:
    def __init__(self, intent: AttackerIntent = AttackerIntent.OPPORTUNISTIC) -> None
    def next_attack_probs(self, current: AttackType) -> np.ndarray
    def next_stage_probs(self, current: KillChainStage) -> np.ndarray
    def escalation_risk(self, current_stage: KillChainStage,
                         probs: np.ndarray | None = None) -> float
    def most_likely_next_stage(self, current: KillChainStage) -> KillChainStage
    def most_likely_next_attack(self, current: AttackType) -> AttackType

# policy_engine.py
class PolicyEngine:
    def __init__(self, model_dir: str = "models/",
                 default_intent: str = "OPPORTUNISTIC") -> None
    def decide(self, state: np.ndarray, features: dict[str, float] | None = None,
               reputation: float = 0.0) -> tuple[HoneypotAction, AttackType | None]
    def decision_info(self, state: np.ndarray) -> dict
    def reload(self, model_dir: str = "models/") -> None
    @property
    def classifier_fitted(self) -> bool
```

## `opencanary_integration.emulator`

```python
# event_generator.py
class OpenCanaryEventGenerator:
    def __init__(self, node_id: str = "honeypot-emulator-01",
                 dst_host: str = "192.168.1.100", seed: int | None = None) -> None
    def generate(self, scenario: str, src_ip: str | None = None) -> OpenCanaryEvent
    def generate_sequence(self, scenarios: list[str], src_ip: str | None = None,
                           delay_ms: float = 0.0) -> list[OpenCanaryEvent]
    def generate_kill_chain(self, src_ip: str | None = None) -> list[OpenCanaryEvent]
    def available_scenarios(self) -> list[str]

# honeypot_emulator.py
class DummyHoneypot:
    def __init__(self, audit_file: str | None = None, verbose: bool = True) -> None
    async def apply_action(self, src_ip, action, attack_type="UNKNOWN",
                            stage="UNKNOWN", threat_level=0.0, event_id="") -> None
    def apply_action_sync(self, src_ip, action, attack_type="UNKNOWN",
                           stage="UNKNOWN", threat_level=0.0, event_id="") -> None
    def get_ip_status(self, src_ip: str) -> dict[str, Any]
    def get_all_sessions(self) -> list[dict[str, Any]]
    def get_action_log(self) -> list[dict[str, Any]]
    def clear_ip(self, src_ip: str) -> bool
    async def schedule_reload(self, urgent: bool = False) -> None
    def schedule_reload_sync(self, urgent: bool = False) -> None
    def close(self) -> None

# scenario.py
class EmulatorScenario:
    def __init__(self, model_dir: str = "models/", intent: str = "OPPORTUNISTIC",
                 audit_file: str | None = None, verbose: bool = True,
                 escalation_mode: str = "window") -> None
    def run_event(self, event: OpenCanaryEvent) -> dict
        # {event_id, src_ip, logtype, service, attack_type, stage, threat_level,
        #  escalation_risk, reputation, action, stage_probs, attack_probs}
```
