"""H1 and H2: frozen now, runnable only after the node has proved itself.

The project owes itself one comparison it has never made. ``CLAUDE.md`` records
that BrickGPT's published training used ``LR 2e-3, r=32, alpha=16`` while the
original plan assumed ``1e-4``/``5e-5`` -- twenty to forty times apart -- and
that **both go into the comparison, with neither assumed better**. These are
those two arms, written down before any number from them exists.

Two things this module is careful about.

**It is a declaration, not a run.** Nothing here imports torch, builds a model
or touches a device. Writing the settings down early is what stops them being
chosen after seeing a result; being unable to execute them is what stops the
declaration turning into a run before the machine has been shown to work.

**The lock is evidence, not an assertion.** :func:`require_unlocked` takes the
six named run directories of the formal gate suite and hands them to
:mod:`src.training.gate_suite`, which re-derives every verdict from the
evidence and requires the six to agree with each other and with the pack
digest, dependency digest, allocator config and determinism settings carried
in separately. It used to take three verdict strings in a dictionary -- a
claim anybody could type, naming no run and no machine.

**Exactly three numbers vary.** ``rank``, ``alpha`` and ``learning_rate``.
Everything else -- dtype, effective batch, sequence length, seed, target
modules, epochs, row count -- is held identical, because an arm that also
changed the batch size would be measuring two things and attributing both to
the learning rate. :func:`differences` recomputes that set from the two
configurations rather than asserting it from memory, so a drifted field shows
up as a failing test instead of as a quietly uncontrolled comparison.

The configurations are :class:`src.training.lora.LoraConfig_` instances. That
type already carries the project's dtype decision and its reasons; a second
config class would be a second place for "bf16, no 4-bit on this project" to
be true, and eventually one of them would stop being.
"""

from __future__ import annotations

from src.training import gate_suite
from src.training.lora import LoraConfig_

#: One epoch over 2,000 rows, per arm. Held here rather than inside the
#: configuration because it describes the *experiment*, not the optimiser.
ROWS = 2000
EPOCHS = 1

#: The six runs that must exist and agree before either arm may be turned
#: into a run. Re-exported from the verifier rather than restated: a second
#: list would be a second place for "which runs count", and the day the two
#: disagreed the looser one would win.
REQUIRED_ROLES = gate_suite.ROLES

#: What the two arms are allowed to differ in. Declared so that
#: :func:`differences` can be checked against an intention rather than merely
#: reported.
VARYING_FIELDS = frozenset({"rank", "alpha", "learning_rate"})


#: The lower-learning-rate arm: the smoke test's settings, which are the
#: project's own prior. ``rank=16, alpha=32`` is a scaling of 2.0.
H1 = LoraConfig_(rank=16, alpha=32, learning_rate=1e-4)

#: BrickGPT's published settings: twenty times the learning rate, twice the
#: rank, and a scaling of 0.5. Not a fallback and not a control -- the
#: published numbers are as much a candidate as ours, which is the whole
#: reason for running both.
H2 = LoraConfig_(rank=32, alpha=16, learning_rate=2e-3)

FROZEN: dict[str, LoraConfig_] = {"H1": H1, "H2": H2}


class HypothesisLocked(RuntimeError):
    """An arm was asked for before the node had earned the right to run it."""


def config_for(name: str) -> LoraConfig_:
    """The frozen configuration for ``name``, or ``KeyError``.

    No default and no fuzzy matching: an arm nobody declared is not an arm,
    and inventing one on a typo is how a third condition appears in a
    two-condition comparison.
    """
    return FROZEN[name]


def differences() -> set[str]:
    """Which fields actually differ, recomputed from the two configurations."""
    a, b = H1.as_dict(), H2.as_dict()
    return {field for field in a if a[field] != b[field]}


def control_problems() -> list[str]:
    """Anything that varies and should not, in one sentence each."""
    unexpected = sorted(differences() - VARYING_FIELDS)
    missing = sorted(VARYING_FIELDS - differences())
    problems = []
    if unexpected:
        problems.append(
            f"{unexpected} differ between H1 and H2 but are supposed to be "
            "held identical; the comparison would attribute their effect to "
            "the learning rate")
    if missing:
        problems.append(
            f"{missing} are declared as the varying fields and are identical "
            "in both arms, so the comparison does not vary what it says it "
            "varies")
    return problems


def require_unlocked(name: str, *, runs, expected_pack_digest,
                     expected_dependency_digest, allocator_config,
                     determinism) -> LoraConfig_:
    """Return the arm's configuration, or refuse and say what is missing.

    The lock used to take three verdict strings in a dictionary. That is a
    claim anybody can type: ``{"gate_8": "passed", ...}`` unlocks both arms
    and names no run, no pack and no machine. It is replaced by the six named
    run directories, which :func:`~src.training.gate_suite.suite_problems`
    re-derives every verdict from and cross-checks against each other and
    against the values carried in.

    Called at the point a run would begin, not at import: the settings stay
    readable, printable and reviewable at any time, and only *executing* them
    is gated. A lock that also hid the numbers would make the frozen design
    impossible to review before it was allowed to run, which is backwards.
    """
    cfg = config_for(name)
    problems = gate_suite.suite_problems(
        runs,
        expected_pack_digest=expected_pack_digest,
        expected_dependency_digest=expected_dependency_digest,
        allocator_config=allocator_config,
        determinism=determinism) + control_problems()
    if problems:
        raise HypothesisLocked(
            f"{name} is frozen and not yet runnable on this node:\n  - "
            + "\n  - ".join(problems))
    return cfg


def summary() -> dict:
    """The declaration, in a form a report can embed verbatim."""
    return {
        "rows": ROWS,
        "epochs": EPOCHS,
        "varying_fields": sorted(VARYING_FIELDS),
        "observed_differences": sorted(differences()),
        "required_roles": list(REQUIRED_ROLES),
        "arms": {name: cfg.as_dict() for name, cfg in FROZEN.items()},
    }
