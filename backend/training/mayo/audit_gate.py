"""
CardioAI Pro - the automatic audit is a required step after automatic labeling
==============================================================================
Every auto label (echo LVEF or MACE outcome) must be run through its automatic audit before anything else can happen
to it. The steps that follow refuse to run without the audit results, and the labels they produce carry a marker that
training data builders check:

  auto-label -> AUTOMATIC AUDIT (required) -> data-scientist audit -> cardiologist -> finalize -> training data

This module holds the shared check. It does not change what the audit decides; a failed check still goes to the
cardiologist. It only makes sure the audit has been run, on every record, before a label can be trained on.
"""
from __future__ import annotations

from typing import Iterable, Optional

VALID = ("pass", "warn", "fail")


class AutoAuditRequired(ValueError):
    """Raised when a step is attempted on labels that have not been through the automatic audit."""


def require_auto_audit(ids: Iterable, results: Optional[dict], what: str = "study") -> None:
    ids = list(ids)
    if not results:
        raise AutoAuditRequired(f"The automatic audit is required after automatic labeling. Run it on the auto labels first "
                                f"(label_auto_audit.auto_audit for echo, mace_label_audit.auto_audit for MACE) and pass the results in.")
    missing = [i for i in ids if i not in results]
    if missing:
        raise AutoAuditRequired(f"The automatic audit has no result for {len(missing)} {what}(s) (e.g. {missing[:3]}). "
                                f"Every auto label must be audited before review.")
    bad = [i for i in ids if results[i].get("status") not in VALID]
    if bad:
        raise AutoAuditRequired(f"Automatic audit result for {len(bad)} {what}(s) has no valid status (e.g. {bad[:3]}).")


def require_plan_audited(plan: dict, what: str = "study") -> None:
    """A routing plan must have been built with the automatic audit (route records each entry's audit status)."""
    missing = [i for i, p in plan.items() if p.get("auto_audit") not in VALID]
    if not plan or missing:
        raise AutoAuditRequired(f"The routing plan was not built with the automatic audit ({len(missing) or 'empty plan'} {what}(s) lack an audit status). "
                                f"Run the automatic audit and pass it to route().")


class AuditedLabels(dict):
    """Final labels that came out of a finalize() whose plan was automatically audited. Plain dicts are not accepted by
    the training-data builders."""
    auto_audited = True


def require_audited_labels(labels) -> None:
    if not getattr(labels, "auto_audited", False):
        raise AutoAuditRequired("These labels did not come from the audited pipeline (automatic audit -> data-scientist audit -> "
                                "cardiologist -> finalize). Training data is built only from finalize() output.")


def audit_counts(plan: dict) -> dict:
    c = {s: sum(1 for p in plan.values() if p.get("auto_audit") == s) for s in VALID}
    return {"studies": len(plan), **c}
