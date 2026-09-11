"""SA-ORL semantic construction core (plan v12).

Implements the language-induced, audited semantic ambiguity set U_alpha
(Algorithm 1) -- the novel object the paper's non-circularity claim rests on.
The offline-RL learner (Algorithm 3) and temporal automaton product (Domain 2)
are deliberately out of scope for this module.
"""
from .construct import ConstructionResult, construct_U_alpha
from .dataset import SemanticDataset
from .dsl import And, Atom, Candidate, Not, Or, Persist, RegionAtom, Since, Within
from .judge import (
    CachedEnsembleJudge,
    DataDrivenJudge,
    LLMEnsembleJudge,
    PlausibilityJudge,
    attach_plausibility,
    load_llm_cache,
)
from .env import ACTIONS, MaintenanceMDP
from .offline import (
    OfflineDataset,
    concat_datasets,
    coverage_behavior,
    epsilon_threshold_behavior,
    evaluate_return,
    exploratory_maintenance_behavior,
    generate_offline_dataset,
    greedy_continue_policy,
    intervention_rate,
    make_offline_rl_dataset,
    policy_costs,
    respect_policy,
    semantic_gap,
    worst_case_cost,
)
from .offline_rl import FQIResult, learn_fqi_constrained
from .neural_rl import learn_cql_constrained
from .cmapss_env import (
    ReplayMDP,
    evaluate_replay_return,
    generate_replay_dataset,
    make_replay_rl_dataset,
)
from .gridworld import (
    CrossingGridworld,
    RULE_SETS_T,
    gridworld_semantic_dataset,
    gw_respect_policy,
    make_gridworld_dataset,
)
from .gridworld_rl import evaluate_gridworld_return, learn_gw_constrained

__all__ = [
    "construct_U_alpha",
    "ConstructionResult",
    "SemanticDataset",
    "Atom",
    "RegionAtom",
    "Not",
    "And",
    "Or",
    "Persist",
    "Within",
    "Since",
    "Candidate",
    "PlausibilityJudge",
    "CachedEnsembleJudge",
    "DataDrivenJudge",
    "LLMEnsembleJudge",
    "attach_plausibility",
    "load_llm_cache",
    "MaintenanceMDP",
    "ACTIONS",
    "OfflineDataset",
    "generate_offline_dataset",
    "epsilon_threshold_behavior",
    "exploratory_maintenance_behavior",
    "coverage_behavior",
    "concat_datasets",
    "make_offline_rl_dataset",
    "greedy_continue_policy",
    "respect_policy",
    "policy_costs",
    "worst_case_cost",
    "intervention_rate",
    "semantic_gap",
    "evaluate_return",
    "learn_fqi_constrained",
    "learn_cql_constrained",
    "FQIResult",
    "ReplayMDP",
    "generate_replay_dataset",
    "make_replay_rl_dataset",
    "evaluate_replay_return",
    "CrossingGridworld",
    "RULE_SETS_T",
    "make_gridworld_dataset",
    "gridworld_semantic_dataset",
    "gw_respect_policy",
    "learn_gw_constrained",
    "evaluate_gridworld_return",
]
