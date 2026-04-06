"""Tests for quantpits.utils.run_context"""
import os
import json
import pytest
from quantpits.utils.run_context import RunContext, _LegacyRunContext, _normalize_script_name


class TestRunContext:
    def test_basic_paths(self, tmp_path):
        ctx = RunContext(
            base_dir=str(tmp_path),
            script_name="brute_force",
            anchor_date="2026-04-03",
        )
        assert ctx.run_dir == str(tmp_path / "brute_force_2026-04-03")
        assert ctx.is_dir == str(tmp_path / "brute_force_2026-04-03" / "is")
        assert ctx.oos_dir == str(tmp_path / "brute_force_2026-04-03" / "oos")

    def test_path_helpers(self, tmp_path):
        ctx = RunContext(
            base_dir=str(tmp_path),
            script_name="minentropy",
            anchor_date="2026-01-01",
        )
        assert ctx.is_path("results.csv").endswith("is/results.csv")
        assert ctx.oos_path("oos_report.txt").endswith("oos/oos_report.txt")
        assert ctx.run_path("run_metadata.json").endswith("minentropy_2026-01-01/run_metadata.json")

    def test_ensure_dirs(self, tmp_path):
        ctx = RunContext(
            base_dir=str(tmp_path),
            script_name="brute_force_fast",
            anchor_date="2026-04-03",
        )
        ctx.ensure_dirs()
        assert os.path.isdir(ctx.run_dir)
        assert os.path.isdir(ctx.is_dir)
        assert os.path.isdir(ctx.oos_dir)

    def test_ensure_dirs_idempotent(self, tmp_path):
        ctx = RunContext(
            base_dir=str(tmp_path),
            script_name="brute_force",
            anchor_date="2026-04-03",
        )
        ctx.ensure_dirs()
        ctx.ensure_dirs()  # Should not raise
        assert os.path.isdir(ctx.run_dir)

    def test_repr(self, tmp_path):
        ctx = RunContext(
            base_dir=str(tmp_path),
            script_name="brute_force",
            anchor_date="2026-04-03",
        )
        r = repr(ctx)
        assert "brute_force" in r
        assert "2026-04-03" in r

    def test_from_metadata_new_structure(self, tmp_path):
        """Test from_metadata with the new per-run directory structure."""
        run_dir = tmp_path / "brute_force_2026-04-03"
        run_dir.mkdir(parents=True)
        (run_dir / "is").mkdir()
        (run_dir / "oos").mkdir()

        meta = {
            "anchor_date": "2026-04-03",
            "script_used": "brute_force_ensemble",
            "freq": "week",
        }
        meta_path = run_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        ctx = RunContext.from_metadata(str(meta_path))
        assert ctx.anchor_date == "2026-04-03"
        assert ctx.script_name == "brute_force"
        assert ctx.run_dir == str(run_dir)
        assert ctx.is_dir == str(run_dir / "is")

    def test_from_metadata_legacy_structure(self, tmp_path):
        """Test from_metadata with old flat directory structure."""
        legacy_dir = tmp_path / "old_output"
        legacy_dir.mkdir()

        meta = {
            "anchor_date": "2026-03-20",
            "script_used": "brute_force_ensemble",
            "freq": "week",
        }
        meta_path = legacy_dir / "run_metadata_2026-03-20.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        ctx = RunContext.from_metadata(str(meta_path))
        assert isinstance(ctx, _LegacyRunContext)
        # Legacy: all dirs point to the same directory
        assert ctx.run_dir == str(legacy_dir)
        assert ctx.is_dir == str(legacy_dir)
        assert ctx.oos_dir == str(legacy_dir)

    def test_from_metadata_fast_script(self, tmp_path):
        """Test from_metadata with brute_force_fast script."""
        run_dir = tmp_path / "brute_force_fast_2026-04-03"
        run_dir.mkdir(parents=True)
        (run_dir / "is").mkdir()
        (run_dir / "oos").mkdir()

        meta = {
            "anchor_date": "2026-04-03",
            "script_used": "brute_force_fast",
            "freq": "week",
        }
        meta_path = run_dir / "run_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        ctx = RunContext.from_metadata(str(meta_path))
        assert ctx.script_name == "brute_force_fast"
        assert "brute_force_fast_2026-04-03" in ctx.run_dir


class TestLegacyRunContext:
    def test_all_dirs_same(self, tmp_path):
        ctx = _LegacyRunContext(
            base_dir=str(tmp_path),
            script_name="brute_force",
            anchor_date="2026-04-03",
            legacy_dir=str(tmp_path / "flat"),
        )
        assert ctx.run_dir == str(tmp_path / "flat")
        assert ctx.is_dir == str(tmp_path / "flat")
        assert ctx.oos_dir == str(tmp_path / "flat")

    def test_ensure_dirs(self, tmp_path):
        flat_dir = str(tmp_path / "flat")
        ctx = _LegacyRunContext(
            base_dir=str(tmp_path),
            script_name="brute_force",
            anchor_date="2026-04-03",
            legacy_dir=flat_dir,
        )
        ctx.ensure_dirs()
        assert os.path.isdir(flat_dir)


class TestNormalizeScriptName:
    def test_known_names(self):
        assert _normalize_script_name("brute_force_ensemble") == "brute_force"
        assert _normalize_script_name("brute_force_fast") == "brute_force_fast"
        assert _normalize_script_name("minentropy") == "minentropy"

    def test_unknown_passthrough(self):
        assert _normalize_script_name("custom_script") == "custom_script"
