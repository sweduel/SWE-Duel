"""Phase 1 tests: configuration loading."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from swe_duel.config import (
    ArenaConfig,
    load_all_repo_configs,
    load_arena_config,
    load_models_config,
    load_repo_config,
)


class TestLoadArenaConfig:
    def test_load_arena_config_valid(self, config_dir: Path, record):
        cfg = load_arena_config(config_dir)
        record("arena_config", cfg.model_dump() if hasattr(cfg, "model_dump") else vars(cfg))
        assert isinstance(cfg, ArenaConfig)
        assert cfg.scoring.lambda_feature == 0.5
        assert cfg.scoring.lambda_bugfix == 0.5
        assert cfg.red_gates.min_lines_added == 5
        assert cfg.sandbox.timeout_seconds == 180
        assert cfg.challenge_bank.target_challenges_per_model_repo == 1
        assert cfg.match.turns_per_player == 3
        assert cfg.rating.elo_k == 32
        # Arena-wide per-response completion cap (uniform across models).
        assert cfg.agent_model.max_tokens == 32768
        # Output-root override lives in arena.yaml (paths.output_dir); the
        # shipped config uses the CWD-relative default.
        assert cfg.paths.output_dir == "data"

    def test_paths_output_dir_validation(self, record):
        from swe_duel.config import PathsConfig

        assert PathsConfig().output_dir == "data"
        assert PathsConfig(output_dir="~/runs").output_dir == "~/runs"
        with pytest.raises(ValueError):
            PathsConfig(output_dir="  ")
        record("default", ArenaConfig().paths.output_dir)

    def test_agent_model_max_tokens_must_be_positive(self, record):
        with pytest.raises(ValidationError) as exc_info:
            ArenaConfig(agent_model={"max_tokens": 0})
        record("error_message", str(exc_info.value))

    def test_load_arena_config_missing_field(self, tmp_path: Path, record):
        bad_yaml = {"scoring": {"lambda_feature": 0.5}}  # missing lambda_bugfix
        cfg_file = tmp_path / "arena.yaml"
        cfg_file.write_text(yaml.dump(bad_yaml))
        cfg = ArenaConfig(**bad_yaml)
        record("input_missing_key", bad_yaml)
        record("resolved_scoring", cfg.scoring.model_dump() if hasattr(cfg.scoring, "model_dump") else vars(cfg.scoring))
        # lambda_bugfix is omitted from the input, so it falls back to the
        # ScoringConfig default (0.5).
        assert cfg.scoring.lambda_bugfix == 0.5

        bad_yaml2 = {"scoring": {"lambda_feature": "not_a_number"}}
        record("input_invalid_type", bad_yaml2)
        with pytest.raises(ValidationError) as exc_info:
            ArenaConfig(**bad_yaml2)
        record("error_message", str(exc_info.value))


class TestLoadModelsConfig:
    """models.yaml carries per-model knobs only; the per-response completion
    cap is arena-wide (arena.yaml ``agent_model.max_tokens``) and is stamped
    onto every entry by the loader."""

    @staticmethod
    def _write_models_yaml(tmp_path: Path, *, with_stale_max_tokens: bool = False) -> Path:
        cfg = tmp_path / "models"
        cfg.mkdir(parents=True, exist_ok=True)
        entry = "    model_id: vendor/m\n    temperature: 0.2\n"
        if with_stale_max_tokens:
            entry += "    max_tokens: 123\n"
        (cfg / "models.yaml").write_text(f"models:\n  m:\n{entry}")
        return cfg

    def test_stamp_applies_arena_max_tokens_to_every_model(self, tmp_path: Path, record):
        cfg_dir = self._write_models_yaml(tmp_path)
        models = load_models_config(cfg_dir, max_tokens=4096)
        record("stamped", {k: v.max_tokens for k, v in models.items()})
        assert models["m"].max_tokens == 4096

    def test_stamp_overrides_stale_per_model_value(self, tmp_path: Path, record):
        cfg_dir = self._write_models_yaml(tmp_path, with_stale_max_tokens=True)
        models = load_models_config(cfg_dir, max_tokens=32768)
        record("stamped", {k: v.max_tokens for k, v in models.items()})
        # One source of truth: the arena-wide value wins over any leftover
        # per-model entry.
        assert models["m"].max_tokens == 32768

    def test_without_stamp_keeps_file_value_or_default(self, tmp_path: Path, record):
        cfg_dir = self._write_models_yaml(tmp_path)
        models = load_models_config(cfg_dir)
        record("unstamped", {k: v.max_tokens for k, v in models.items()})
        # No per-model value and no stamp: the ModelConfig default (mirroring
        # arena.yaml's agent_model.max_tokens) applies.
        assert models["m"].max_tokens == 32768

    def test_real_config_files_agree(self, config_dir: Path, record):
        arena = load_arena_config(config_dir)
        models = load_models_config(config_dir, max_tokens=arena.agent_model.max_tokens)
        record("max_tokens", {k: v.max_tokens for k, v in models.items()})
        assert models
        assert all(v.max_tokens == arena.agent_model.max_tokens for v in models.values())


class TestLoadRepoConfig:
    def test_load_repo_config_no_modules(self, config_dir: Path, record):
        flask_path = config_dir / "repos" / "flask.yaml"
        rc = load_repo_config(flask_path)
        record("repo_config", rc.model_dump() if hasattr(rc, "model_dump") else vars(rc))
        assert rc.name == "flask"
        assert rc.url == "https://github.com/pallets/flask"
        assert rc.commit == "3.1.1"
        assert rc.docker_image == "swe-duel-flask"
        # Ensure no 'modules' field exists in the YAML
        with open(flask_path) as f:
            raw = yaml.safe_load(f)
        assert "modules" not in raw.get("repo", {})


class TestLoadAllRepoConfigs:
    def test_load_all_repo_configs(self, config_dir: Path, record):
        repos = load_all_repo_configs(config_dir)
        record("repos", {k: (v.model_dump() if hasattr(v, "model_dump") else vars(v)) for k, v in repos.items()})
        # flask, jinja, sqlalchemy (python); helmet, expressjs, node-jsonwebtoken
        # (node); jwt, chi, csrf (go); java-html-sanitizer, java-jwt, jjwt
        # (java); cjson, libexpat, simdjson (c).
        assert len(repos) == 15
        assert "flask" in repos
        assert "jinja" in repos
        assert "sqlalchemy" in repos
        assert "helmet" in repos
        assert "jwt" in repos
        assert "java-html-sanitizer" in repos
        assert "cjson" in repos
        assert "expressjs" in repos
        assert "chi" in repos
        assert "libexpat" in repos
        assert "java-jwt" in repos
        assert "csrf" in repos
        assert "node-jsonwebtoken" in repos
        assert "jjwt" in repos
        assert "simdjson" in repos

    def test_non_python_repo_configs(self, config_dir: Path, record):
        """Per-language repos declare their language + native test command."""
        repos = load_all_repo_configs(config_dir)

        helmet = repos["helmet"]
        record("helmet", helmet.model_dump() if hasattr(helmet, "model_dump") else vars(helmet))
        assert helmet.language == "node"
        assert helmet.docker_image == "swe-duel-helmet"
        assert helmet.url == "https://github.com/helmetjs/helmet"
        assert "tsx --test" in helmet.test_command

        expressjs = repos["expressjs"]
        record("expressjs", expressjs.model_dump() if hasattr(expressjs, "model_dump") else vars(expressjs))
        assert expressjs.language == "javascript"
        assert expressjs.docker_image == "swe-duel-expressjs"
        assert expressjs.url == "https://github.com/expressjs/express"
        assert "mocha" in expressjs.test_command

        jwt = repos["jwt"]
        record("jwt", jwt.model_dump() if hasattr(jwt, "model_dump") else vars(jwt))
        assert jwt.language == "go"
        assert jwt.docker_image == "swe-duel-jwt"
        assert jwt.url == "https://github.com/golang-jwt/jwt"
        assert "go test" in jwt.test_command

        chi = repos["chi"]
        record("chi", chi.model_dump() if hasattr(chi, "model_dump") else vars(chi))
        assert chi.language == "go"
        assert chi.docker_image == "swe-duel-chi"
        assert chi.url == "https://github.com/go-chi/chi"
        assert "go test" in chi.test_command

        java_sanitizer = repos["java-html-sanitizer"]
        record("java_html_sanitizer", java_sanitizer.model_dump() if hasattr(java_sanitizer, "model_dump") else vars(java_sanitizer))
        assert java_sanitizer.language == "java"
        assert java_sanitizer.docker_image == "swe-duel-java-html-sanitizer"
        assert java_sanitizer.url == "https://github.com/owasp/java-html-sanitizer"
        assert "mvn test" in java_sanitizer.test_command

        java_jwt = repos["java-jwt"]
        record("java_jwt", java_jwt.model_dump() if hasattr(java_jwt, "model_dump") else vars(java_jwt))
        assert java_jwt.language == "java"
        assert java_jwt.docker_image == "swe-duel-java-jwt"
        assert java_jwt.url == "https://github.com/auth0/java-jwt"
        assert "gradlew" in java_jwt.test_command

        cjson = repos["cjson"]
        record("cjson", cjson.model_dump() if hasattr(cjson, "model_dump") else vars(cjson))
        assert cjson.language == "c"
        assert cjson.docker_image == "swe-duel-cjson"
        assert cjson.url == "https://github.com/davegamble/cjson"
        assert "make test" in cjson.test_command

        libexpat = repos["libexpat"]
        record("libexpat", libexpat.model_dump() if hasattr(libexpat, "model_dump") else vars(libexpat))
        assert libexpat.language == "c"
        assert libexpat.docker_image == "swe-duel-libexpat"
        assert libexpat.url == "https://github.com/libexpat/libexpat"
        assert "cmake" in libexpat.test_command
        assert "ctest" in libexpat.test_command

        csrf = repos["csrf"]
        record("csrf", csrf.model_dump() if hasattr(csrf, "model_dump") else vars(csrf))
        assert csrf.language == "go"
        assert csrf.docker_image == "swe-duel-csrf"
        assert csrf.url == "https://github.com/gorilla/csrf"
        assert "go test" in csrf.test_command

        sqlalchemy = repos["sqlalchemy"]
        record(
            "sqlalchemy",
            sqlalchemy.model_dump() if hasattr(sqlalchemy, "model_dump") else vars(sqlalchemy),
        )
        assert sqlalchemy.language == "python"
        assert sqlalchemy.docker_image == "swe-duel-sqlalchemy"
        assert sqlalchemy.url == "https://github.com/sqlalchemy/sqlalchemy"
        assert "pytest" in sqlalchemy.test_command

        node_jwt = repos["node-jsonwebtoken"]
        record(
            "node_jsonwebtoken",
            node_jwt.model_dump() if hasattr(node_jwt, "model_dump") else vars(node_jwt),
        )
        assert node_jwt.language == "javascript"
        assert node_jwt.docker_image == "swe-duel-node-jsonwebtoken"
        assert node_jwt.url == "https://github.com/auth0/node-jsonwebtoken"
        assert "mocha" in node_jwt.test_command

        jjwt = repos["jjwt"]
        record("jjwt", jjwt.model_dump() if hasattr(jjwt, "model_dump") else vars(jjwt))
        assert jjwt.language == "java"
        assert jjwt.docker_image == "swe-duel-jjwt"
        assert jjwt.url == "https://github.com/jwtk/jjwt"
        assert "mvn" in jjwt.test_command

        simdjson = repos["simdjson"]
        record(
            "simdjson",
            simdjson.model_dump() if hasattr(simdjson, "model_dump") else vars(simdjson),
        )
        assert simdjson.language == "c"
        assert simdjson.docker_image == "swe-duel-simdjson"
        assert simdjson.url == "https://github.com/simdjson/simdjson"
        assert "cmake" in simdjson.test_command
        assert "ctest" in simdjson.test_command


class TestConfigDefaults:
    """The packaged templates (swe_duel.config_defaults) must track config/.

    `swe-duel init` copies them into fresh working directories, and
    `swe-duel setup`/`swe-duel doctor` rely on the template repo set being complete,
    so a repo added to config/repos/ without updating the package templates
    would silently ship an incomplete wheel.
    """

    def test_repo_templates_match_live_config(self, config_dir: Path, record):
        import swe_duel.config_defaults as defaults_pkg

        defaults_dir = Path(defaults_pkg.__file__).resolve().parent
        live = load_all_repo_configs(config_dir)
        bundled = load_all_repo_configs(defaults_dir)
        record("live_repos", sorted(live))
        record("bundled_repos", sorted(bundled))
        assert set(bundled) == set(live)
        for name, rc in bundled.items():
            assert rc.url == live[name].url, name
            assert rc.commit == live[name].commit, name

    def test_templates_parse_standalone(self, record):
        import swe_duel.config_defaults as defaults_pkg

        from swe_duel.config import load_arena_config, load_models_config

        defaults_dir = Path(defaults_pkg.__file__).resolve().parent
        arena = load_arena_config(defaults_dir)
        models = load_models_config(
            defaults_dir, max_tokens=arena.agent_model.max_tokens
        )
        record("n_models", len(models))
        assert models, "template models.yaml must declare at least one model"
