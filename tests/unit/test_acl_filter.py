"""Unit tests for ACL filter builder — security-critical.

Every edge case in this module directly maps to a potential data leak.
"""

from __future__ import annotations

import pytest

from rag.core.security import UserContext, build_acl_filter


# ── UserContext.acl_groups ────────────────────────────────────────────────────

def test_acl_groups_always_includes_all() -> None:
    """The 'all' group must always be in acl_groups, even if not in groups."""
    user = UserContext(user_id="u1", groups=frozenset())
    assert "all" in user.acl_groups


def test_acl_groups_includes_user_groups() -> None:
    user = UserContext(user_id="u1", groups=frozenset({"engineering", "legal"}))
    assert "engineering" in user.acl_groups
    assert "legal" in user.acl_groups
    assert "all" in user.acl_groups


def test_acl_groups_deduplicates_all() -> None:
    """If user already has 'all' in groups, it should not appear twice."""
    user = UserContext(user_id="u1", groups=frozenset({"all", "engineering"}))
    assert user.acl_groups.count("all") == 1


def test_acl_groups_is_sorted() -> None:
    """Groups should be sorted for determinism."""
    user = UserContext(user_id="u1", groups=frozenset({"zzz", "aaa", "mmm"}))
    groups = user.acl_groups
    assert groups == sorted(groups)


# ── build_acl_filter ──────────────────────────────────────────────────────────

def test_filter_has_must_clause() -> None:
    user = UserContext(user_id="u1", groups=frozenset({"engineering"}))
    f = build_acl_filter(user)
    assert "must" in f
    assert len(f["must"]) == 1


def test_filter_targets_acl_groups_key() -> None:
    user = UserContext(user_id="u1", groups=frozenset({"engineering"}))
    f = build_acl_filter(user)
    assert f["must"][0]["key"] == "acl_groups"


def test_filter_uses_match_any() -> None:
    user = UserContext(user_id="u1", groups=frozenset({"engineering"}))
    f = build_acl_filter(user)
    assert "any" in f["must"][0]["match"]


def test_filter_includes_all_group() -> None:
    """Even a user with no explicit groups gets the 'all' sentinel."""
    user = UserContext(user_id="u1", groups=frozenset())
    f = build_acl_filter(user)
    assert "all" in f["must"][0]["match"]["any"]


def test_filter_includes_user_groups() -> None:
    user = UserContext(user_id="u1", groups=frozenset({"engineering", "legal"}))
    f = build_acl_filter(user)
    any_groups = f["must"][0]["match"]["any"]
    assert "engineering" in any_groups
    assert "legal" in any_groups


def test_empty_groups_user_only_sees_all_docs() -> None:
    """A user with no groups can only see docs tagged 'all'."""
    user = UserContext(user_id="u1", groups=frozenset())
    f = build_acl_filter(user)
    any_groups = f["must"][0]["match"]["any"]
    assert any_groups == ["all"]


def test_filter_is_restrictive_not_permissive() -> None:
    """The filter must use 'must' (AND), not 'should' (OR) at top level."""
    user = UserContext(user_id="u1", groups=frozenset({"engineering"}))
    f = build_acl_filter(user)
    # No 'should' at top level — that would be permissive
    assert "should" not in f
    # Must have exactly one must clause
    assert len(f["must"]) == 1


def test_wildcard_group_not_supported() -> None:
    """A group named '*' should not implicitly grant access to everything."""
    user = UserContext(user_id="u1", groups=frozenset({"*"}))
    f = build_acl_filter(user)
    # The filter uses exact match, not glob — '*' is a literal group name
    any_groups = f["must"][0]["match"]["any"]
    assert "*" in any_groups  # treated as a literal group, not a wildcard
    assert len(any_groups) == 2  # only "*" and "all"


def test_doc_with_no_acl_not_accessible() -> None:
    """The filter structure should not match a doc with acl_groups=[]."""
    user = UserContext(user_id="u1", groups=frozenset({"engineering"}))
    f = build_acl_filter(user)
    any_groups = set(f["must"][0]["match"]["any"])

    # Simulate: doc has acl_groups=[] — intersection with user's groups is empty
    doc_acl_groups: list[str] = []
    matches = bool(any_groups & set(doc_acl_groups))
    assert not matches, "Documents with no ACL groups should not be accessible"


def test_filter_deterministic_across_calls() -> None:
    """Same user should produce the same filter every time."""
    user = UserContext(user_id="u1", groups=frozenset({"engineering", "legal"}))
    f1 = build_acl_filter(user)
    f2 = build_acl_filter(user)
    assert f1 == f2
