"""Report-format and reproducibility-package contract tests."""
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from package_project import REQUIRED_CODE, package_project
from report import (add_adaptive_findings, add_exception_findings, add_joint_findings,
                    add_joint_convergence_findings, add_negative_findings, add_final_field_comparisons,
                    build_content, build_final_content, build_multi_final_content,
                    find_fonts, generate_report, validate_content)


ROOT = Path(__file__).resolve().parents[1]


def evidence_path(name):
    """Work in the source tree and in an extracted ZIP with code/ beside evidence/."""
    candidates = [ROOT / "evidence" / name, ROOT.parent / "evidence" / name]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def short_content():
    return {"schema_version": 1, "status": "DRAFT", "group": "99", "members": [],
            "title": "Formatting fixture", "evidence_label": "Synthetic fixture, not research results",
            "cover_notes": [], "source_sha256": {}, "sections": [
                {"task": task, "title": f"Task {task}", "paragraphs": ["Synthetic test content."],
                 "tables": [{"caption": "Fixture", "columns": ["Name", "Value"], "rows": [["Check", "1"]]}],
                 "discussion": "This fixture checks formatting only."} for task in [1, 2, 3]]}


def adaptive_fixture(directory):
    """Schema fixture only, never research evidence."""
    directory.mkdir()
    protocol = {"schema_version": 1, "test_read": False, "stage": "development", "seeds": [99],
                "training_objective": "categorical_cross_entropy", "selection_metric": "meta_fit_macro_cross_entropy",
                "rating_categories": [1, 2, 3, 4, 5], "ranking_adapter": "P(rating>=4)",
                "checkpoints": [10, 30, 60, 100]}
    models = {name: {"categorical": {"macro_cross_entropy": 1.1, "macro_brier": .5,
                                     "macro_expected_rating_rmse": .8},
                     "ranking": {"liked_ratings": {"ndcg@10": .2}, "all_observed": {"ndcg@10": .15}}}
              for name in ["adaptive", "fixed_flow", "hard_clamp", "global_histogram", "item_histogram", "item_user_product"]}
    aggregates = {"test_read": False, "seeds": {"99": {"models": models}}}
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "aggregates.json").write_text(json.dumps(aggregates))
    return protocol, aggregates


def joint_fixture(directory):
    """Distinct readout values make accidental endpoint interchange observable."""
    directory.mkdir()
    protocol = {"schema_version": 1, "test_read": False, "stage": "development", "seeds": [99],
                "training_objective": "joint_recorded_item_rating_multinomial",
                "selection_metric": "meta_fit_macro_joint_nll", "rating_categories": [1, 2, 3, 4, 5],
                "ranking_adapters": {"all_observed": "logsumexp(all5logits)",
                                     "liked_record": "logsumexp(rating4,5logits)"},
                "checkpoints": [10, 30, 60, 100]}
    models = {name: {"joint": {"macro_joint_nll": 7.1}, "categorical": {"macro_cross_entropy": 1.2},
                     "ranking": {"all_observed_adapter": {"all_observed": {"ndcg@10": .31},
                                                            "liked_ratings": {"ndcg@10": .12}},
                                 "liked_record_adapter": {"all_observed": {"ndcg@10": .09},
                                                          "liked_ratings": {"ndcg@10": .42}}}}
              for name in ["adaptive", "fixed_flow"]}
    aggregates = {"test_read": False, "seeds": {"99": {"models": models}}}
    (directory / "protocol.json").write_text(json.dumps(protocol))
    (directory / "aggregates.json").write_text(json.dumps(aggregates))
    return protocol, aggregates


def convergence_fixture(directory):
    protocol, aggregates = joint_fixture(directory)
    protocol.update(study_kind="post_v1_convergence_sensitivity", checkpoints=[10, 30, 60, 100, 200, 300, 400])
    (directory / "protocol.json").write_text(json.dumps(protocol))
    return protocol, aggregates


def frozen_field_fixture(base, make_fixture):
    """Construct metadata-only final outputs; no test labels or model files exist."""
    source, output, frozen = base / "source", base / (base.name + "-final"), base / "frozen"
    protocol, aggregates = make_fixture(source)
    output.mkdir()
    frozen.mkdir()
    aggregates.update({"schema_version": 1, "stage": "test", "test_read": True})
    bundle_hashes = {}
    for seed, result in aggregates["seeds"].items():
        directory = frozen / seed
        directory.mkdir()
        per_seed_protocol = {**protocol, "seeds": [int(seed)], "source_run": f"private/local/path/{seed}"}
        (directory / "protocol.json").write_text(json.dumps(per_seed_protocol))
        protocol_hash = hashlib.sha256((directory / "protocol.json").read_bytes()).hexdigest()
        result["selections"] = {name: {"epoch": 60} for name in ["adaptive", "fixed_flow"]}
        bundle = {"status": "frozen", "seed": int(seed), "k": 10, "test_read": False, "test_evaluated": False,
                  "models": list(result["models"]), "selections": result["selections"],
                  "source_protocol_sha256": protocol_hash, "payload_sha256": {"protocol.json": protocol_hash}}
        (directory / "bundle.json").write_text(json.dumps(bundle))
        bundle_hashes[seed] = hashlib.sha256((directory / "bundle.json").read_bytes()).hexdigest()
    seal = {"format_version": 1, "status": "frozen", "test_read": False, "test_evaluated": False,
            "seeds": list(map(int, aggregates["seeds"])), "bundle_sha256": bundle_hashes,
            "code_sha256": {"synthetic.py": "synthetic source hash"}}
    (frozen / "manifest.json").write_text(json.dumps(seal))
    seal_hash = hashlib.sha256((frozen / "manifest.json").read_bytes()).hexdigest()
    (frozen / "manifest.sha256").write_text(seal_hash + "\n")
    (output / "aggregates.json").write_text(json.dumps(aggregates))
    manifest = {"status": "complete", "test_read": True, "test_evaluated": True, "selection_after_test": False,
                "frozen_manifest_sha256": seal_hash, "code_sha256": seal["code_sha256"],
                "output_sha256": {"aggregates.json": hashlib.sha256((output / "aggregates.json").read_bytes()).hexdigest()}}
    (output / "manifest.json").write_text(json.dumps(manifest))
    return output, frozen, manifest


def primary_final_fixture(base):
    """Synthetic final-format fixtures from development aggregates, never test data."""
    evidence = evidence_path("research-v3")
    aggregates = json.loads((evidence / "aggregates.json").read_text())
    selected = json.loads((evidence / "selected.json").read_text())
    provenance = json.loads((evidence / "provenance.json").read_text())
    evaluations = []
    for seed in ["2026", "2027", "2028"]:
        output, frozen = base / (seed + "-results"), base / (seed + "-frozen")
        output.mkdir()
        frozen.mkdir()
        models = {name: {"kind": "fixture"} for name in aggregates[seed]}
        sources = provenance[seed]["manifest"]["sources"]
        for name in sources:
            models[name] = {"kind": "expert", "expert": name}
        families = {family: row["name"] for family, row in selected[seed].items() if family != "expert"}
        coefficients = json.loads((evidence / (seed + "-coefficients.json")).read_text())
        for family, name in families.items():
            models[name] = {"kind": "linear", "variant": "static" if family in {"constrained", "calibrated"}
                            else family.removesuffix("-pairwise"), "coefficients": coefficients[name]}
        bundle = {"status": "frozen", "test_read": False, "seed": int(seed), "models": models, "sources": sources,
                  "selection": {"families": families, "best_expert": selected[seed]["expert"]["name"]},
                  "code_sha256": {"synthetic.py": "a" * 64},
                  "source_artifacts_sha256": {name: {"manifest.json": "b" * 64} for name in sources},
                  "expected_test_sha256": hashlib.sha256(seed.encode()).hexdigest()}
        (frozen / "freeze.json").write_text(json.dumps(bundle))
        (output / "results.json").write_text(json.dumps(aggregates[seed]))
        manifest = {"status": "complete", "test_read": True, "test_model_selection": False,
                    "test_sha256": bundle["expected_test_sha256"],
                    "k": 10, "users": 17, "interactions": 99, "models": list(models), "protocol": {},
                    "freeze_sha256": hashlib.sha256((frozen / "freeze.json").read_bytes()).hexdigest(),
                    "output_sha256": {"results.json": hashlib.sha256((output / "results.json").read_bytes()).hexdigest()}}
        (output / "manifest.json").write_text(json.dumps(manifest))
        evaluations.append((seed, output, frozen))
    return evaluations


def field_comparison_fixture(base, content):
    """Complete synthetic metadata only, with endpoint-specific values and primary references."""
    seeds = [2026, 2027, 2028]
    tracks = [("conditional", adaptive_fixture, add_adaptive_findings),
              ("joint100", joint_fixture, add_joint_findings),
              ("joint400", convergence_fixture, add_joint_convergence_findings)]
    for track, maker, reader in tracks:
        directory = base / track
        directory.mkdir()
        def three_seeds(path, maker=maker):
            protocol, aggregate = maker(path)
            protocol["seeds"] = seeds
            aggregate["seeds"] = {str(seed): json.loads(json.dumps(aggregate["seeds"]["99"])) for seed in seeds}
            (path / "protocol.json").write_text(json.dumps(protocol))
            (path / "aggregates.json").write_text(json.dumps(aggregate))
            return protocol, aggregate
        output, frozen, _ = frozen_field_fixture(directory, three_seeds)
        reader(content, output, frozen)
    paired = []
    for seed in seeds:
        comparisons = {}
        for track, _, _ in tracks:
            for endpoint in ["all_observed", "liked_ratings"]:
                candidate = (.15 if endpoint == "all_observed" else .2) if track == "conditional" else (.31 if endpoint == "all_observed" else .42)
                for reference in ["fixed_flow", "EASE", "SLIMElastic", "PositiveEASE"]:
                    other = candidate if reference == "fixed_flow" else .41
                    if endpoint == "all_observed" and reference in {"EASE", "SLIMElastic"}:
                        role = content["canonical_roles"][str(seed)]["expert:" + reference]
                        other = content["aggregate_evidence"][str(seed)][role]["aggregate"]["ndcg@10"]
                    key = f"{track}/{endpoint}/adaptive-minus-{reference}"
                    comparisons[key] = {"track": track, "endpoint": endpoint, "reference": reference,
                        "candidate_metrics": {"ndcg@10": candidate}, "reference_metrics": {"ndcg@10": other},
                        "mean_ndcg_difference": candidate - other,
                        "bonferroni_family_interval_95": [candidate - other - .1, candidate - other + .1]}
        paired.append({"seed": seed, "comparisons": comparisons})
    rows = {}
    for key, first in paired[0]["comparisons"].items():
        values = [row["comparisons"][key] for row in paired]
        rows[key] = {key: first[key] for key in ["track", "endpoint", "reference"]}
        rows[key].update(seeds=3, mean_candidate_ndcg=sum(row["candidate_metrics"]["ndcg@10"] for row in values) / 3,
                         mean_reference_ndcg=sum(row["reference_metrics"]["ndcg@10"] for row in values) / 3,
                         mean_ndcg_difference=sum(row["mean_ndcg_difference"] for row in values) / 3)
    directory = base / "comparisons"
    directory.mkdir()
    aggregate = {"seeds": seeds, "comparisons": rows, "cross_seed_confidence_interval": None}
    (directory / "aggregate.json").write_text(json.dumps(aggregate))
    (directory / "per-seed-comparisons.json").write_text(json.dumps(paired))
    manifest = {"status": "complete", "selection_performed": False, "comparisons_per_seed": 24, "seeds": seeds,
                "inputs": {track: {key: content["supplemental_evidence"][study]["final_evaluation"][key]
                                   for key in ["frozen_manifest_sha256", "evaluation_manifest_sha256"]}
                           for track, study in [("conditional", "adaptive"), ("joint100", "joint"),
                                                ("joint400", "joint-convergence")]},
                "output_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                  for name in ["aggregate.json", "per-seed-comparisons.json"]}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, aggregate, manifest


class ReportTests(unittest.TestCase):
    def test_discussion_limit_and_task_coverage(self):
        content = short_content()
        content["sections"][0]["discussion"] = " ".join(["word"] * 200)
        self.assertEqual(validate_content(content)["1"], 200)
        content["sections"][0]["discussion"] += " overflow"
        with self.assertRaisesRegex(ValueError, "exceeds 200"):
            validate_content(content)
        content = short_content()
        content["sections"].pop()
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            validate_content(content)

    def test_default_evidence_is_described_as_development(self):
        content = build_content(evidence_path("research-v2"))
        self.assertIsNone(content["group"])
        self.assertEqual(content["members"], [])
        self.assertEqual(content["status"], "DRAFT")
        self.assertTrue(all(n <= 200 for n in validate_content(content).values()))
        expected = json.loads((evidence_path("research-v2") / "selected.json").read_text())
        mean = sum(v["expert"]["ndcg@10"] for v in expected.values()) / len(expected)
        paired = content["sections"][0]["tables"][0]["rows"]
        entries = [row[i:i + 2] for row in paired for i in [0, 2]]
        self.assertIn(["EASE", f"{mean:.4f}"], entries)

    def test_model_table_uses_source_types_and_actual_cohort_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "evidence"
            shutil.copytree(evidence_path("research-v2"), evidence)
            provenance = json.loads((evidence / "provenance.json").read_text())
            aggregates = json.loads((evidence / "aggregates.json").read_text())
            selected = json.loads((evidence / "selected.json").read_text())
            for seed in provenance:
                name = seed + "-SyntheticExtra-0"
                provenance[seed]["manifest"]["sources"][name] = {"manifest": {"model": "SyntheticExtra"}}
                provenance[seed]["manifest"]["development_users"] = 236
                provenance[seed]["manifest"]["calibration_users"] = 236
                aggregates[seed][name] = aggregates[seed][selected[seed]["expert"]["name"]]
                path = evidence / (seed + "-coefficients.json")
                coefficients = json.loads(path.read_text())
                coefficients[selected[seed]["static"]["name"]]["weights"][name] = .125
                path.write_text(json.dumps(coefficients))
            (evidence / "provenance.json").write_text(json.dumps(provenance))
            (evidence / "aggregates.json").write_text(json.dumps(aggregates))
            content = build_content(evidence)
            self.assertIn("236 development users", " ".join(content["sections"][0]["paragraphs"]))
            self.assertIn("SyntheticExtra", json.dumps(content["sections"][0]["tables"]))
            self.assertIn("+0.1250", json.dumps(content["sections"][1]["tables"]))

    def test_exception_findings_are_separate_and_preserve_negative_result(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            values = {name + ":liked_ratings": {"like_ndcg": value, "known_dislike_rate_per_slot": .01}
                      for name, value in [("contrast_transfer", .1), ("signed_channels", .3),
                                          ("positive_ease", .25), ("pair_gate", .09)]}
            (directory / "summary.json").write_text(json.dumps({"synthetic": values}))
            content = add_exception_findings(build_content(evidence_path("research-v2")), directory)
            self.assertIn("failed this comparison", content["sections"][1]["discussion"])
            self.assertIn("not directly comparable", " ".join(content["sections"][1]["paragraphs"]))
            self.assertTrue(all(n <= 200 for n in validate_content(content).values()))

    def test_adaptive_appendix_keeps_main_results_and_rejects_wrong_endpoint_or_records(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "adaptive-fixture"
            protocol, aggregates = adaptive_fixture(directory)
            content = short_content()
            main_before = json.dumps(content["sections"], sort_keys=True)
            add_adaptive_findings(content, directory)
            self.assertEqual(json.dumps(content["sections"], sort_keys=True), main_before)
            appendix = content["appendices"][0]
            self.assertIn("does not improve", appendix["discussion"])
            self.assertIn("distinct from the assignment", " ".join(appendix["paragraphs"]))
            self.assertIn("10/30/60/100", " ".join(appendix["paragraphs"]))
            self.assertLessEqual(validate_content(content)["appendix-adaptive"], 200)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                add_adaptive_findings(content, directory)
            protocol["rating_categories"] = [0, 1]
            (directory / "protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "five-category"):
                add_adaptive_findings(short_content(), directory)
            protocol["rating_categories"] = [1, 2, 3, 4, 5]
            (directory / "protocol.json").write_text(json.dumps(protocol))
            aggregates["seeds"]["99"]["per_user"] = {"private-id": 1}
            (directory / "aggregates.json").write_text(json.dumps(aggregates))
            with self.assertRaisesRegex(ValueError, "Individual records"):
                add_adaptive_findings(short_content(), directory)

    def test_joint_objective_readouts_remain_distinct_and_combine_only_matching_seeds(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            adaptive_fixture(base / "conditional")
            protocol, aggregates = joint_fixture(base / "joint")
            content = add_adaptive_findings(short_content(), base / "conditional")
            add_joint_findings(content, base / "joint")
            self.assertEqual(len(content["appendices"]), 1)
            appendix = content["appendices"][0]
            self.assertEqual(appendix["tables"][1]["rows"][0],
                             ["Joint adaptive", "7.1000", "1.2000", "0.3100", "0.4200"])
            self.assertEqual(set(content["supplemental_evidence"]), {"adaptive", "joint"})
            self.assertIn("different sample spaces", appendix["discussion"])
            self.assertLessEqual(validate_content(content)["appendix-joint"], 200)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                add_joint_findings(content, base / "joint")
            protocol["seeds"] = [98]
            aggregates["seeds"]["98"] = aggregates["seeds"].pop("99")
            (base / "joint/protocol.json").write_text(json.dumps(protocol))
            (base / "joint/aggregates.json").write_text(json.dumps(aggregates))
            with self.assertRaisesRegex(ValueError, "matching completed seed sets"):
                add_joint_findings(add_adaptive_findings(short_content(), base / "conditional"), base / "joint")

    def test_negative_appendix_reports_primary_and_secondary_without_replacing_main(self):
        content = short_content()
        add_negative_findings(content, evidence_path("negative-information-v2"))
        appendix = content["appendices"][0]
        self.assertEqual(appendix["tables"][0]["rows"][0], ["Positive only", "0.1618", "0.1813", "0.1734", "0.1860"])
        self.assertIn("-0.0009", appendix["discussion"])
        self.assertEqual(len(content["sections"]), 3)
        self.assertLessEqual(validate_content(content)["appendix-negative"], 200)
        self.assertIn("code/NEGATIVE_INFORMATION_PROTOCOL.md", content["source_sha256"])
        appendix["discussion"] = "word " * 201
        with self.assertRaisesRegex(ValueError, "exceeds 200"):
            validate_content(content)
        with tempfile.TemporaryDirectory() as temp:
            wrong_protocol = Path(temp) / "protocol.md"
            wrong_protocol.write_text("altered protocol")
            with self.assertRaisesRegex(ValueError, "protocol hash mismatch"):
                add_negative_findings(short_content(), evidence_path("negative-information-v2"), wrong_protocol)

    def test_final_field_readers_verify_seals_without_reading_labels_and_keep_stage_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "conditional").mkdir()
            (root / "joint").mkdir()
            conditional, frozen_c, manifest = frozen_field_fixture(root / "conditional", adaptive_fixture)
            joint, frozen_j, _ = frozen_field_fixture(root / "joint", joint_fixture)
            original_open = Path.open
            def metadata_only(path, *args, **kwargs):
                self.assertIn(path.name, {"manifest.json", "manifest.sha256", "aggregates.json", "bundle.json", "protocol.json"})
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", metadata_only):
                content = add_adaptive_findings(short_content(), conditional, frozen_c)
                add_joint_findings(content, joint, frozen_j)
            appendix = content["appendices"][0]
            self.assertEqual(appendix["evidence_stage"], "held-out test")
            self.assertNotIn("test stays closed", json.dumps(appendix))
            self.assertIn("TRAIN+validation", json.dumps(appendix))
            self.assertNotIn("private/local/path", json.dumps(content))
            self.assertEqual(len(appendix["tables"]), 2)
            self.assertIn("conditional-final/manifest.json", content["source_sha256"])
            # Post-test choices and corrupted aggregate bytes cannot be labelled final.
            manifest["selection_after_test"] = True
            (conditional / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "post-test selection"):
                add_adaptive_findings(short_content(), conditional, frozen_c)
            manifest["selection_after_test"] = False
            (conditional / "manifest.json").write_text(json.dumps(manifest))
            (conditional / "aggregates.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "aggregate hash mismatch"):
                add_adaptive_findings(short_content(), conditional, frozen_c)

    def test_convergence_is_separate_and_cannot_replace_original_joint_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            joint_fixture(root / "joint")
            convergence_fixture(root / "convergence")
            with self.assertRaisesRegex(ValueError, "explicit convergence"):
                add_joint_findings(short_content(), root / "convergence")
            with self.assertRaisesRegex(ValueError, "original joint-field"):
                add_joint_convergence_findings(short_content(), root / "convergence")
            content = add_joint_findings(short_content(), root / "joint")
            original = json.dumps(content["appendices"][0], sort_keys=True)
            add_joint_convergence_findings(content, root / "convergence")
            self.assertEqual(json.dumps(content["appendices"][0], sort_keys=True), original)
            appendix = content["appendices"][1]
            self.assertEqual(appendix["tables"][0]["rows"][0][0], "100ep adaptive")
            self.assertEqual(appendix["tables"][0]["rows"][2][0], "400ep adaptive")
            self.assertIn("not independent confirmation", " ".join(appendix["paragraphs"]))
            self.assertLessEqual(validate_content(content)["appendix-joint-convergence"], 200)

    def test_final_reader_checks_freeze_hash_and_removes_individual_metrics(self):
        evidence = evidence_path("research-v2")
        results = json.loads((evidence / "aggregates.json").read_text())["2026"]
        selected = json.loads((evidence / "selected.json").read_text())["2026"]
        coefficients = json.loads((evidence / "2026-coefficients.json").read_text())
        static = selected["static"]["name"]
        specs = {name: {"kind": "expert" if name.startswith("2026-") and "/" not in name else "fixture"}
                 for name in results}
        specs[static] = {"kind": "linear", "coefficients": coefficients[static]}
        bundle = {"models": specs, "selection": {"families": {name: row["name"] for name, row in selected.items()}}}
        results[static]["per_user"] = {"PRIVATE_USER_ID": {"metric": 1}}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "freeze.json").write_text(json.dumps(bundle))
            (directory / "results.json").write_text(json.dumps(results))
            manifest = {"status": "complete", "test_read": True, "test_model_selection": False,
                        "k": 10, "users": 943, "interactions": 10000, "models": list(results), "protocol": {},
                        "freeze_sha256": hashlib.sha256((directory / "freeze.json").read_bytes()).hexdigest(),
                        "output_sha256": {"results.json": hashlib.sha256((directory / "results.json").read_bytes()).hexdigest()}}
            (directory / "manifest.json").write_text(json.dumps(manifest))
            content = build_final_content(directory, directory)
            self.assertNotIn("PRIVATE_USER_ID", json.dumps(content))
            self.assertEqual(content["status"], "DRAFT")
            self.assertTrue(all(n <= 200 for n in validate_content(content).values()))
            content["status"] = "FINAL"
            with self.assertRaisesRegex(ValueError, "five distinct"):
                validate_content(content)
            content["group"] = "99"
            content["members"] = [f"Synthetic fixture contributor {i}" for i in range(5)]
            validate_content(content)
            (directory / "results.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                build_final_content(directory, directory)

    def test_multi_primary_final_preserves_all_manifests_and_rejects_missing_or_swapped_roles(self):
        with tempfile.TemporaryDirectory() as temp:
            evaluations = primary_final_fixture(Path(temp))
            content = build_multi_final_content(evaluations, "99")
            self.assertEqual(set(content["final_evaluation"]["seeds"]), {"2026", "2027", "2028"})
            self.assertEqual(set(content["aggregate_evidence"]), {"2026", "2027", "2028"})
            self.assertEqual(set(content["canonical_roles"]), {"2026", "2027", "2028"})
            entries = [row[i:i+2] for row in content["sections"][0]["tables"][0]["rows"] for i in [0, 2]]
            expected = json.loads((evidence_path("research-v3") / "individual-models.json").read_text())
            mean = sum(seed["EASE"]["aggregate"]["ndcg@10"] for seed in expected.values()) / 3
            self.assertIn(["EASE", f"{mean:.4f}"], entries)
            self.assertEqual(sum(row[0] == "EASE" for row in entries), 1)
            self.assertEqual([row[0] for row in content["sections"][1]["tables"][1]["rows"]], ["EASE", "SLIMElastic", "context"])
            with self.assertRaisesRegex(ValueError, "complete primary seed set"):
                build_multi_final_content(evaluations[:2])
            _, output, frozen = evaluations[0]
            bundle = json.loads((frozen / "freeze.json").read_text())
            families = bundle["selection"]["families"]
            families["static"], families["context"] = families["context"], families["static"]
            (frozen / "freeze.json").write_text(json.dumps(bundle))
            manifest = json.loads((output / "manifest.json").read_text())
            manifest["freeze_sha256"] = hashlib.sha256((frozen / "freeze.json").read_bytes()).hexdigest()
            (output / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "family role"):
                build_multi_final_content(evaluations)

    def test_final_field_comparisons_match_frozen_roles_and_keep_liked_endpoint_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            content = build_multi_final_content(primary_final_fixture(root), "99")
            directory, aggregate, manifest = field_comparison_fixture(root, content)
            add_final_field_comparisons(content, directory)
            discussion = content["sections"][1]["discussion"]
            self.assertIn("0.310 at the 100-epoch cap and 0.310 at the 400-epoch cap", discussion)
            self.assertIn("joint400 scores 0.420", discussion)
            self.assertIn("PositiveEASE 0.410", discussion)
            self.assertIn("favor adaptive in 0/3 splits and fixed flow in 0/3", discussion)
            self.assertLessEqual(validate_content(content)["2"], 200)
            original_hash = manifest["inputs"]["joint100"]["evaluation_manifest_sha256"]
            manifest["inputs"]["joint100"]["evaluation_manifest_sha256"] = "mismatched exact run"
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different frozen evaluation"):
                add_final_field_comparisons(content, directory)
            manifest["inputs"]["joint100"]["evaluation_manifest_sha256"] = original_hash
            aggregate["comparisons"]["joint400/all_observed/adaptive-minus-EASE"]["mean_reference_ndcg"] = .999
            (directory / "aggregate.json").write_text(json.dumps(aggregate))
            manifest["output_sha256"]["aggregate.json"] = hashlib.sha256((directory / "aggregate.json").read_bytes()).hexdigest()
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "disagree"):
                add_final_field_comparisons(content, directory)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") and importlib.util.find_spec("pypdf"),
                         "PDF verification requires matplotlib and pypdf")
    def test_multi_primary_final_main_pages_fit_at_required_font(self):
        try:
            find_fonts()
        except FileNotFoundError:
            self.skipTest("Times New Roman fonts not available")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            content = build_multi_final_content(primary_final_fixture(root), "99")
            manifest = generate_report(content, root / "report")
            self.assertEqual(manifest["layout"]["pages"], 4)
            self.assertTrue(all(count <= 200 for count in manifest["discussion_words"].values()))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") and importlib.util.find_spec("pypdf"),
                         "PDF verification requires matplotlib and pypdf")
    def test_complete_final_comparison_report_fits_seven_pages(self):
        try:
            find_fonts()
        except FileNotFoundError:
            self.skipTest("Times New Roman fonts not available")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            content = build_multi_final_content(primary_final_fixture(root), "99")
            directory, _, _ = field_comparison_fixture(root, content)
            add_final_field_comparisons(content, directory)
            add_negative_findings(content, evidence_path("negative-information-v2"))
            manifest = generate_report(content, root / "report")
            self.assertEqual(manifest["layout"]["pages"], 7)
            self.assertTrue(all(bottom > 61.8 for bottom in manifest["layout"]["page_bottoms_pt"]))

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") and importlib.util.find_spec("pypdf"),
                         "PDF verification requires matplotlib and pypdf")
    def test_pdf_four_pages_embeds_tnr_and_is_deterministic(self):
        try:
            find_fonts()
        except FileNotFoundError:
            self.skipTest("Times New Roman fonts not available")
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a", Path(temp) / "b"
            content = short_content()
            manifest = generate_report(content, first)
            generate_report(content, second)
            self.assertEqual((first / "report.pdf").read_bytes(), (second / "report.pdf").read_bytes())
            reader = PdfReader(first / "report.pdf")
            self.assertEqual(len(reader.pages), 4)
            self.assertTrue(all("DRAFT" in p.extract_text() for p in reader.pages))
            fonts = [f.get_object() for p in reader.pages for f in p["/Resources"]["/Font"].values()]
            self.assertTrue(any("TimesNewRoman" in str(f.get("/BaseFont", "")) for f in fonts))
            self.assertEqual(manifest["layout"]["font_size"], 12)
            self.assertEqual(manifest["layout"]["line_spacing"], 1.15)
            with self.assertRaises(FileExistsError):
                generate_report(content, first)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Requires PDF renderer")
    def test_page_overflow_is_rejected_without_font_shrink(self):
        try:
            find_fonts()
        except FileNotFoundError:
            self.skipTest("Times New Roman fonts not available")
        content = short_content()
        content["sections"][0]["paragraphs"] = ["Long content. " * 2000]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "overflows"):
                generate_report(content, Path(temp) / "report")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib") and importlib.util.find_spec("pypdf"),
                         "PDF verification requires matplotlib and pypdf")
    def test_supplemental_pdf_preserves_three_task_pages_and_separate_page_limits(self):
        try:
            find_fonts()
        except FileNotFoundError:
            self.skipTest("Times New Roman fonts not available")
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            adaptive_fixture(directory / "adaptive-fixture")
            joint_fixture(directory / "joint-fixture")
            convergence_fixture(directory / "convergence-fixture")
            content = add_adaptive_findings(short_content(), directory / "adaptive-fixture")
            add_joint_findings(content, directory / "joint-fixture")
            add_joint_convergence_findings(content, directory / "convergence-fixture")
            add_negative_findings(content, evidence_path("negative-information-v2"))
            manifest = generate_report(content, directory / "report")
            reader = PdfReader(directory / "report/report.pdf")
            self.assertEqual(len(reader.pages), 7)
            self.assertEqual(manifest["layout"]["main_task_pages"], 3)
            self.assertEqual(manifest["layout"]["appendix_pages"], 3)
            self.assertIn("Supplemental development | 5/7", reader.pages[4].extract_text())
            self.assertIn("Supplemental development | 6/7", reader.pages[5].extract_text())
            self.assertTrue(all(bottom > 61.8 for bottom in manifest["layout"]["page_bottoms_pt"]))


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.report = self.base / "code", self.base / "report"
        self.root.mkdir()
        self.report.mkdir()
        for name in REQUIRED_CODE:
            (self.root / name).write_text("fixture\n")
        for name in ["report.pdf", "report.md", "report-content.json"]:
            (self.report / name).write_text("fixture " + name)
        self.manifest = {"status": "DRAFT", "group": "99", "contributors_supplied": False,
                         "source_sha256": {}, "artifact_sha256": {
                             name: hashlib.sha256((self.report / name).read_bytes()).hexdigest()
                             for name in ["report.pdf", "report.md", "report-content.json"]}}
        self.write_manifest()

    def write_manifest(self):
        (self.report / "REPORT-MANIFEST.json").write_text(json.dumps(self.manifest))

    def test_explicit_whitelist_hashes_and_determinism(self):
        for name in ["runs", "dataset", "reference"]:
            (self.root / name).mkdir()
            (self.root / name / "private.txt").write_text("excluded")
        (self.root / ".env").write_text("excluded")
        (self.root / "unknown.py").write_text("excluded")
        a, b = self.base / "a/99.zip", self.base / "b/99.zip"
        package_project(self.root, self.report, a, "99", allow_draft=True)
        package_project(self.root, self.report, b, "99", allow_draft=True)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        with zipfile.ZipFile(a) as archive:
            manifest = json.loads(archive.read("PACKAGE-MANIFEST.json"))
            for name, expected in manifest["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
            self.assertEqual(set(archive.namelist()), set(manifest["artifact_sha256"]) | {"PACKAGE-MANIFEST.json"})
            self.assertFalse(any(part in " ".join(archive.namelist()) for part in ["private", ".env", "unknown.py"]))
        with self.assertRaises(FileExistsError):
            package_project(self.root, self.report, a, "99", allow_draft=True)

    def test_draft_tampering_group_and_symlink_rejections(self):
        out = self.base / "99.zip"
        with self.assertRaisesRegex(ValueError, "DRAFT"):
            package_project(self.root, self.report, out, "99")
        with self.assertRaisesRegex(ValueError, "Group number differs"):
            package_project(self.root, self.report, self.base / "98.zip", "98", allow_draft=True)
        (self.report / "report.pdf").write_text("changed after manifest")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            package_project(self.root, self.report, out, "99", allow_draft=True)
        (self.report / "report.pdf").write_text("fixture report.pdf")
        (self.root / "README.md").unlink()
        (self.root / "README.md").symlink_to(self.report / "report.md")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            package_project(self.root, self.report, out, "99", allow_draft=True)

    def test_matching_report_sources_required(self):
        self.manifest["source_sha256"] = {"selected.json": "wrong"}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "matching aggregate evidence"):
            package_project(self.root, self.report, self.base / "99.zip", "99", allow_draft=True)

    def test_allowed_aggregate_filename_cannot_hide_individual_ids(self):
        evidence = self.base / "evidence"
        evidence.mkdir()
        (evidence / "group-policies.json").write_text(json.dumps({
            "2026": {"independent_calibration": {"calibration_users": ["private-id"]}}}))
        with self.assertRaisesRegex(ValueError, "identifier lists"):
            package_project(self.root, self.report, self.base / "99.zip", "99", [evidence], allow_draft=True)

    def test_supplemental_source_protocol_hash_and_aggregate_whitelist(self):
        evidence = self.base / "negative-fixture"
        evidence.mkdir()
        protocol = self.root / "NEGATIVE_INFORMATION_PROTOCOL.md"
        protocol.write_text("synthetic protocol")
        (self.root / "categorical_field.py").write_text("# synthetic source\n")
        aggregate = evidence / "aggregate.json"
        aggregate.write_text('{"count": 3}')
        (evidence / "raw-ratings.json").write_text('{"private-id": 1}')
        self.manifest["source_sha256"] = {
            "code/NEGATIVE_INFORMATION_PROTOCOL.md": hashlib.sha256(protocol.read_bytes()).hexdigest(),
            "negative-fixture/aggregate.json": hashlib.sha256(aggregate.read_bytes()).hexdigest()}
        self.write_manifest()
        out = self.base / "99.zip"
        package_project(self.root, self.report, out, "99", [evidence], allow_draft=True)
        with zipfile.ZipFile(out) as archive:
            self.assertIn("code/categorical_field.py", archive.namelist())
            self.assertIn("evidence/negative-fixture/aggregate.json", archive.namelist())
            self.assertFalse(any("raw-ratings" in name for name in archive.namelist()))
        protocol.write_text("changed protocol")
        with self.assertRaisesRegex(ValueError, "matching aggregate evidence"):
            package_project(self.root, self.report, self.base / "other/99.zip", "99", [evidence], allow_draft=True)

    def test_optional_demo_is_synthetic_source_bound_and_copies_only_four_files(self):
        demo = self.base / "demo"
        demo.mkdir()
        for name in ["categorical_demo.py", "categorical_field.py"]:
            (self.root / name).write_text("# synthetic test source\n")
        provenance = {"model_source_sha256": hashlib.sha256((self.root / "categorical_field.py").read_bytes()).hexdigest()}
        (demo / "index.html").write_text("<p>Synthetic fixture only</p>")
        (demo / "predictions.json").write_text(json.dumps({"profile_kind": "hand_authored_synthetic",
                                                          "provenance": provenance, "history": []}))
        (demo / "validation.json").write_text(json.dumps({"status": "PASS", "scope": "synthetic fixture"}))
        (demo / "unapproved.json").write_text('{"private-id": 1}')
        manifest = {"profile_kind": "hand_authored_synthetic", "real_user_history_read": False, "test_read": False,
                    "provenance": provenance, "generator_sha256": hashlib.sha256((self.root / "categorical_demo.py").read_bytes()).hexdigest(),
                    "files_sha256": {name: hashlib.sha256((demo / name).read_bytes()).hexdigest()
                                     for name in ["index.html", "predictions.json"]}}
        (demo / "manifest.json").write_text(json.dumps(manifest))
        out = self.base / "99.zip"
        package_project(self.root, self.report, out, "99", allow_draft=True, demo_dir=demo)
        with zipfile.ZipFile(out) as archive:
            self.assertEqual({name for name in archive.namelist() if name.startswith("demo/")},
                             {"demo/" + name for name in ["index.html", "predictions.json", "manifest.json", "validation.json"]})
            self.assertTrue(json.loads(archive.read("PACKAGE-MANIFEST.json"))["demo_included"])
        (demo / "index.html").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Demo file hash mismatch"):
            package_project(self.root, self.report, self.base / "tampered/99.zip", "99", allow_draft=True, demo_dir=demo)
        manifest["real_user_history_read"] = True
        (demo / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "synthetic profile"):
            package_project(self.root, self.report, self.base / "private/99.zip", "99", allow_draft=True, demo_dir=demo)


if __name__ == "__main__":
    unittest.main()
