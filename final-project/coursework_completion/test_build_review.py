"""Packaging boundaries and portable links, using synthetic public artifacts."""
import json
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from coursework_completion import build_review as builder


def synthetic_completion():
    protocol = {"seeds": [2026, 2027, 2028], "grids": builder.GRIDS, "candidate_count": 39,
                "roles": list(builder.ROLES), "original_test_read": False, "fresh_holdout": False,
                "assessment_stage": "exploratory reused validation calibration cohort"}
    data = {"status": "complete", "original_test_read": False, "fresh_holdout": False,
            "assessment_stage": protocol["assessment_stage"], "seeds": {}, "mean_metrics": {}}
    for seed_index, seed in enumerate(builder.SEEDS):
        candidates, selections = [], {}
        for family in builder.FAMILIES:
            key = {"mixed": "quotas", "meta": "lambda", "tuned_switch": "group_count", "tuned_rrf": "offset"}[family]
            for index, setting in enumerate(builder.GRIDS[family]):
                candidate = {"family": family, "candidate_id": f"{family}-{index}", "settings": {key: setting},
                             "development": {"users": 236, "aggregate": {"ndcg@10": .2+index/100}}}
                candidates.append(candidate)
            selections[family] = candidates[-1]
        assessed = {role: {"users": 236, "positive_pairs": 1500,
            "aggregate": {metric: .1+role_index/100+seed_index/1000
                          for metric in ("ndcg@10", "recall@10", "mrr@10")}}
            for role_index, role in enumerate(builder.ROLES)}
        data["seeds"][seed] = {"cohort_sizes": {"meta_fit": 471, "development": 236, "calibration": 236},
            "candidates": candidates, "selections": selections, "assessment": assessed,
            "references": {role: {} for role in builder.ROLES[:5]}}
    data["mean_metrics"] = {role: {metric: sum(data["seeds"][seed]["assessment"][role]["aggregate"][metric]
        for seed in builder.SEEDS)/3 for metric in ("ndcg@10", "recall@10", "mrr@10")} for role in builder.ROLES}
    return data, protocol


class ReviewPackagingTests(unittest.TestCase):
    def test_equal_split_means_canonical_roles_and_grid_selection(self):
        data, protocol = synthetic_completion()
        means = builder.validate_completion(data, protocol)
        self.assertAlmostEqual(means["mixed"]["ndcg@10"], .151)
        self.assertAlmostEqual(means["EASE"]["ndcg@10"], .101)
        for change in ("missing_seed", "role_swap", "changed_cohort", "wrong_winner"):
            altered = copy.deepcopy(data)
            if change == "missing_seed":
                del altered["seeds"]["2028"]
            elif change == "role_swap":
                rows = altered["seeds"]["2026"]["assessment"]
                rows["mixed"], rows["EASE"] = rows["EASE"], rows["mixed"]
            elif change == "changed_cohort":
                altered["seeds"]["2026"]["assessment"]["mixed"]["users"] = 235
            else:
                altered["seeds"]["2026"]["selections"]["mixed"] = altered["seeds"]["2026"]["candidates"][0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                builder.validate_completion(altered, protocol)

    def test_completion_panel_preserves_original_scientific_tables_and_scope(self):
        roots = (builder.ROOT/"reports/framing-review-v1/main-report-content.json",
                 builder.ROOT.parent/"archive/framing-review/main-report-content.json")
        path = next((path for path in roots if path.is_file()), None)
        if path is None:
            self.skipTest("Historical public report fixture not available")
        original = json.loads(path.read_text())
        before = copy.deepcopy(original)
        data, protocol = synthetic_completion()
        completion = {"data": data, "means": builder.validate_completion(data, protocol),
                      "sources": {}, "payload": {}, "audit_payload": {}}
        result = builder.build_main_content(original, completion)
        self.assertEqual(original, before)
        self.assertEqual(result["aggregate_evidence"], original["aggregate_evidence"])
        self.assertEqual(result["sections"][0]["tables"][0], original["sections"][0]["tables"][0])
        self.assertEqual(result["sections"][0]["tables"][1]["rows"][0],
                         ["Mixed lists", "0.1510", "Meta-level", "0.1610"])
        self.assertIn("Separate exploratory", result["sections"][0]["tables"][1]["caption"])
        self.assertIn("societal reranking (cascade)", " ".join(result["sections"][0]["paragraphs"]))
        self.assertIn("236 previously exposed", " ".join(result["sections"][0]["paragraphs"]))
        self.assertIn("constrained weights sum to one", " ".join(result["sections"][0]["paragraphs"]))
        self.assertLessEqual(len(result["sections"][0]["discussion"].split()), 200)
        self.assertEqual(result["sections"][2]["tables"], original["sections"][2]["tables"])
        policy_discussion = result["sections"][2]["discussion"]
        self.assertIn("TRAIN-activity group, independent calibration selects exposure strength", policy_discussion)
        self.assertIn("minimize the catalog-share gap", policy_discussion)
        self.assertIn("nonnegative Bonferroni-bootstrap lower bound on mean nDCG minus 95%", policy_discussion)
        self.assertIn("not a held-out guarantee", policy_discussion)
        self.assertLessEqual(len(policy_discussion.split()), 200)
        if importlib.util.find_spec("matplotlib") is None:
            return
        import report
        try:
            report.find_fonts()
        except FileNotFoundError:
            return
        with tempfile.TemporaryDirectory() as temporary:
            layout = report.render_pdf(result, Path(temporary)/"synthetic-layout.pdf")
        self.assertEqual(layout["pages"], 4)
        self.assertGreater(min(layout["page_bottoms_pt"]), 53)

    def test_navigation_maps_to_archive_and_labels_unbundled_history(self):
        original = ("[current](reports/coursework-complete-v1/report.pdf) "
                    "[completed](reports/coursework-complete-v2/report.pdf) "
                    "[old](reports/framing-review-v1/report.pdf#page=2) "
                    "[evidence](evidence/example/aggregates.json) "
                    "[external](https://example.org/a) "
                    "[missing](exploratory/unbundled/RESULTS.md)").encode()
        payload = {"report.pdf": b"pdf", "archive/framing-review/report.pdf": b"old",
                   "evidence/example/aggregates.json": b"{}"}
        output, changes = builder.portable_document(original, "README.md", payload)
        text = output.decode()
        self.assertIn("[current](../report.pdf)", text)
        self.assertIn("[completed](../report.pdf)", text)
        self.assertIn("[old](../archive/framing-review/report.pdf#page=2)", text)
        self.assertIn("[evidence](../evidence/example/aggregates.json)", text)
        self.assertIn("[external](https://example.org/a)", text)
        self.assertIn("missing (source-checkout reference:", text)
        self.assertEqual(len(changes), 5)
        payload["code/README.md"] = output
        self.assertEqual(builder.check_document_links(payload, ["code/README.md"])["unresolved_internal_links"], 0)
        self.assertNotEqual(original, output)

    def test_link_check_rejects_missing_relative_targets(self):
        with self.assertRaisesRegex(ValueError, "Broken packaged navigation"):
            builder.check_document_links({"code/README.md": b"[absent](missing.md)"}, ["code/README.md"])

    def test_public_evidence_requires_exact_hashes_and_rejects_private_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            content = {"aggregates.json": {"users": 236, "metric": .25},
                       "protocol.json": {"test_read": False}, "provenance.json": {"status": "complete"}}
            for name, value in content.items():
                (directory/name).write_bytes(builder.json_bytes(value))
            index = {name: builder.digest(directory/name) for name in content}
            (directory/"SHA256.json").write_bytes(builder.json_bytes(index))
            self.assertEqual(set(builder.checked_public(directory)), set(content)|{"SHA256.json"})
            (directory/"aggregates.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                builder.checked_public(directory)
            (directory/"aggregates.json").write_bytes(builder.json_bytes({"per_user": {"private": .5}}))
            index["aggregates.json"] = builder.digest(directory/"aggregates.json")
            (directory/"SHA256.json").write_bytes(builder.json_bytes(index))
            with self.assertRaisesRegex(ValueError, "Individual data excluded"):
                builder.checked_public(directory)

    def test_source_closure_includes_local_import_and_refuses_changed_base_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"entry.py").write_text("from helper import value\n")
            (root/"helper.py").write_text("value = 1\n")
            payload = {}
            files = builder.scientific_source_payload(payload, ["entry.py"], {}, root)
            self.assertEqual(files, {"entry.py", "helper.py"})
            self.assertEqual(payload["code/helper.py"], b"value = 1\n")
            (root/"helper.py").write_text("value = 2\n")
            with self.assertRaisesRegex(ValueError, "scientific source changed"):
                builder.scientific_source_payload(payload, ["entry.py"], {}, root)

    def test_completion_audit_binds_exact_sources_barrier_and_public_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/builder.STUDY).mkdir()
            (root/builder.STUDY/"models.py").write_text("# scientific fixture\n")
            (root/builder.STUDY/"verify.py").write_text("# independent fixture\n")
            public, audited = root/"public", root/"audit"
            public.mkdir()
            audited.mkdir()
            data, protocol = synthetic_completion()
            sources = {builder.STUDY+"/models.py": builder.digest(root/builder.STUDY/"models.py")}
            protocol.update(source_sha256=sources, runtime={"python": "fixture"})
            payload = {"aggregates.json": builder.json_bytes(data), "protocol.json": builder.json_bytes(protocol)}
            provenance = {"status": "complete", "original_test_read": False, "fresh_holdout": False,
                "assessment_after_all_seed_selections": True, "protocol_sha256": builder.sha(payload["protocol.json"]),
                "aggregate_sha256": builder.sha(payload["aggregates.json"]), "source_sha256": sources,
                "runtime": protocol["runtime"], "selection_seal_sha256": "a"*64,
                "output_sha256": {"SELECTIONS-FROZEN.json": "a"*64}}
            payload["provenance.json"] = builder.json_bytes(provenance)
            hashes = {name: builder.sha(raw) for name, raw in payload.items()}
            for name, raw in payload.items():
                (public/name).write_bytes(raw)
            (public/"SHA256.json").write_bytes(builder.json_bytes(hashes))
            verifier = builder.digest(root/builder.STUDY/"verify.py")
            audit = {"status": "passed", "scientific_public_artifacts": hashes,
                "scientific_source_sha256": sources, "verifier_source_sha256": verifier,
                "selection_barrier_sha256": "a"*64, "assessment_manifest_sha256": hashes["provenance.json"],
                "maximum_absolute_error": 1e-14, "checks_completed": 50}
            def write_audit(value):
                (audited/"audit.json").write_bytes(builder.json_bytes(value))
                (audited/"SHA256.json").write_bytes(builder.json_bytes({
                    "audit.json": builder.digest(audited/"audit.json"), "verify.py": verifier}))
            write_audit(audit)
            result = builder.completion_evidence(public, audited, root)
            self.assertEqual(result["audit_payload"]["verify.py"], b"# independent fixture\n")
            for key in ("selection_barrier_sha256", "assessment_manifest_sha256", "verifier_source_sha256"):
                bad = {**audit, key: "b"*64}
                write_audit(bad)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    builder.completion_evidence(public, audited, root)
            write_audit(audit)
            (root/builder.STUDY/"models.py").write_text("# changed scientific fixture\n")
            with self.assertRaises(ValueError):
                builder.completion_evidence(public, audited, root)

    def test_reproduction_payload_includes_recipe_and_safe_receipts_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/builder.STUDY
            (source/"reproduction").mkdir(parents=True)
            (source/"rebuild_recipe.json").write_text('{"seeds": [2026, 2027, 2028]}')
            (source/"reproduction/replay.json").write_text('{"status": "passed", "users": 943}')
            (source/"reproduction/private.npy").write_bytes(b"not included")
            payload = {}
            snapshot = builder.reproduction_payload(payload, root)
            self.assertEqual(len(snapshot), 2)
            self.assertIn("code/"+builder.STUDY+"/rebuild_recipe.json", payload)
            self.assertFalse(any(name.endswith(".npy") for name in payload))
            (source/"reproduction/replay.json").write_text('{"per_user": {"private": 1}}')
            with self.assertRaisesRegex(ValueError, "Individual data excluded"):
                builder.reproduction_payload({}, root)

    def test_historical_root_artifacts_are_preserved_without_overwriting_nested_evidence(self):
        original = {"report.pdf": b"prior pdf", "README.md": b"prior readme",
                    "code/example.py": b"example", "evidence/results.json": b"aggregate",
                    "archive/original-coursework/report.pdf": b"original"}
        copied = builder.initial_payload(original)
        self.assertEqual(copied["archive/framing-review/report.pdf"], original["report.pdf"])
        self.assertEqual(copied["archive/original-coursework/report.pdf"], b"original")
        self.assertEqual(copied["evidence/results.json"], b"aggregate")
        self.assertEqual(original["report.pdf"], b"prior pdf")
        self.assertNotIn("report.pdf", copied)

    def test_quick_check_payload_uses_fixed_public_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"operations").mkdir()
            for name in ("operations/__init__.py", "operations/team_smoke_check.py", "requirements-smoke.txt"):
                (root/name).write_text("# public fixture\n")
            (root/"operations/private.json").write_text('{"private": true}')
            payload = {}
            snapshot = builder.operational_payload(payload, root)
            self.assertEqual(set(snapshot), {"operations/__init__.py", "operations/team_smoke_check.py",
                                             "requirements-smoke.txt"})
            self.assertEqual(set(payload), {"code/"+name for name in snapshot})

    def test_navigation_includes_current_team_docs_but_not_historical_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ("README.md", "HANDOFF.md", "PLAN.md", "REPRODUCE.md", "docs/COMPLETION.md",
                     "coursework_completion/README.md", "coursework_completion/COVERAGE.md",
                     "coursework_completion/TEAM_REVIEW.md", "docs/PROGRESS.md",
                     "docs/team-meeting-2026-10-01/QUICKSTART.md")
            for name in names:
                (root/name).parent.mkdir(parents=True, exist_ok=True)
                (root/name).write_text("# public navigation\n")
            (root/"README.md").write_text("[progress](docs/PROGRESS.md) [done](docs/COMPLETION.md)")
            (root/"docs/archive").mkdir()
            (root/"docs/archive/private.md").write_text("excluded history\n")
            payload = {}
            _, documents, snapshot = builder.add_navigation(payload, root)
            self.assertEqual(set(snapshot), set(names))
            self.assertEqual(set(documents), {"code/"+name for name in names})
            self.assertEqual(builder.check_document_links(payload, documents)["unresolved_internal_links"], 0)

    def test_unverified_base_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"24.zip"
            path.write_bytes(b"not the verified archive")
            with self.assertRaisesRegex(ValueError, "verified framing"):
                builder.checked_base(path)
        for name in ("../x", "/private/x", "runs/history.json", "scores.npy", "weights.pt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.safe_name(name)


if __name__ == "__main__":
    unittest.main()
