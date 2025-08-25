import math
import json
import re
import itertools
from pathlib import Path
import pandas as pd
from .paths import get_data_dir
from .huggingface import download_dataset


class DataDecidePaths:
    def __init__(self):
        self.data_dir = get_data_dir() / "datadecide"
        self.dataset_dir = self.data_dir / "datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.ds_details_path = self.data_dir / "dataset_features.csv"

    def parquet_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.parquet"

    def dataset_path(self, max_params_str: str) -> Path:
        return self.dataset_dir / f"dataset_{max_params_str}.pkl"

    # ------------ DataDecide Raw Paths ------------
    @property
    def ppl_eval_raw_path(self) -> Path:
        return self.parquet_path("ppl_eval")

    @property
    def downstream_eval_raw_path(self) -> Path:
        return self.parquet_path("downstream_eval")

    # ------------ DataDecide Parsed Paths ------------
    @property
    def step_to_token_compute_path(self) -> Path:
        return self.parquet_path("step_to_token_compute")

    @property
    def ppl_eval_parsed_path(self) -> Path:
        return self.parquet_path("ppl_eval_parsed")

    @property
    def downstream_eval_parsed_path(self) -> Path:
        return self.parquet_path("downstream_eval_parsed")

    @property
    def full_eval_ds_path(self) -> Path:
        return self.parquet_path("full_eval")

    @property
    def mean_eval_ds_path(self) -> Path:
        return self.parquet_path("mean_eval")

    @property
    def std_eval_ds_path(self) -> Path:
        return self.parquet_path("std_eval")


# Model details are from https://github.com/allenai/OLMo/blob/7094aab0448096f4994cae881edbd629d6c2f3de/scripts/ladder.py#L352
class DataDecideDefaults:
    def __init__(self):
        self.model_configs = {}

        self.hf_ds_names = {
            "downstream_eval_ds": "allenai/DataDecide-eval-results",
            "downstream_instance_ds": "allenai/DataDecide-eval-instances",
            "perplexity_eval_ds": "allenai/DataDecide-ppl-results",
        }
        self._number_unit_re = re.compile(r"^([0-9]+)([a-zA-Z]+)$")
        self.max_seq_len = 2_048
        self.model_size_norm_value = 108_000_000
        self.lr_exponent = -1 / 3
        self.lr_max_base = 0.0047
        self.lr_final_ratio = 0.01
        self.bs_exponent = 2 / 3
        self.gpus_per_node = 8  # used for bs rounding
        self.microbatch_size = 4  # used for bs rounding
        self.model_shapes = {
            "4M": {"d_model": 64, "n_heads": 8, "n_layers": 8, "mlp_ratio": 8},
            "6M": {"d_model": 96, "n_heads": 8, "n_layers": 8, "mlp_ratio": 8},
            "8M": {"d_model": 128, "n_heads": 8, "n_layers": 8, "mlp_ratio": 8},
            "10M": {"d_model": 144, "n_heads": 8, "n_layers": 8, "mlp_ratio": 8},
            "14M": {"d_model": 192, "n_heads": 8, "n_layers": 8, "mlp_ratio": 8},
            "16M": {"d_model": 208, "n_heads": 8, "n_layers": 8, "mlp_ratio": 8},
            "20M": {"d_model": 192, "n_heads": 8, "n_layers": 16, "mlp_ratio": 8},
            "60M": {"d_model": 384, "n_heads": 12, "n_layers": 16, "mlp_ratio": 8},
            "90M": {"d_model": 528, "n_heads": 12, "n_layers": 16, "mlp_ratio": 8},
            "150M": {"d_model": 768, "n_heads": 12, "n_layers": 12, "mlp_ratio": 8},
            "300M": {"d_model": 1024, "n_heads": 16, "n_layers": 16, "mlp_ratio": 8},
            "530M": {"d_model": 1344, "n_heads": 16, "n_layers": 16, "mlp_ratio": 8},
            "750M": {"d_model": 1536, "n_heads": 16, "n_layers": 16, "mlp_ratio": 8},
            "1B": {"d_model": 2048, "n_heads": 16, "n_layers": 16, "mlp_ratio": 8},
        }
        self.hardcoded_size_mapping = {
            "4M": 3_744_832,
            "6M": 6_010_464,
            "8M": 8_538_240,
            "10M": 9_900_432,
            "14M": 14_380_224,
            "16M": 16_004_560,
            "20M": 19_101_888,
            "60M": 57_078_144,
            "90M": 97_946_640,
            "150M": 150_000_000,
            "300M": 300_000_000,
            "530M": 530_000_000,
            "750M": 750_000_000,
            "1B": 1_000_000_000,
        }
        self.ppl_name_map = {
            "eval/wikitext_103-validation/Perplexity": "wikitext_103-valppl",
            "eval/pile-validation/Perplexity": "pile-valppl",
            "eval/c4_en-validation/Perplexity": "c4_en-valppl",
            "eval/m2d2_s2orc-validation/Perplexity": "m2d2_s2orc-valppl",
            "eval/ice-validation/Perplexity": "ice-valppl",
            "eval/dolma_wiki-validation/Perplexity": "dolma_wiki-valppl",
            "eval/dolma_stack-validation/Perplexity": "dolma_stack-valppl",
            "eval/dolma_reddit-validation/Perplexity": "dolma_reddit-valppl",
            "eval/dolma_pes2o-validation/Perplexity": "dolma_pes2o-valppl",
            "eval/dolma_common-crawl-validation/Perplexity": "dolma_common-crawl-valppl",
            "eval/dolma_books-validation/Perplexity": "dolma_books-valppl",
        }
        self.seed_map = {
            "default": 0,
            "small aux 2": 1,
            "small aux 3": 2,
            "large aux 2": 3,
            "large aux 3": 4,
        }
        self.data_recipe_families = {
            "dolma17": [
                "Dolma1.7",
                "Dolma1.7 (no code)",
                "Dolma1.7 (no math, code)",
                "Dolma1.7 (no Reddit)",
                "Dolma1.7 (no Flan)",
            ],
            "dolma16pp": ["Dolma1.6++"],
            "c4": ["C4"],
            "fineweb": ["FineWeb-Pro", "FineWeb-Edu"],
            "falcon": ["Falcon"],
            "falcon_cc": [
                "Falcon+CC",
                "Falcon+CC (QC 10%)",
                "Falcon+CC (QC 20%)",
                "Falcon+CC (QC Orig 10%)",
                "Falcon+CC (QC Tulu 10%)",
            ],
            "dclm": [
                "DCLM-Baseline",
                "DCLM-Baseline (QC 10%)",
                "DCLM-Baseline (QC 20%)",
                "DCLM-Baseline (QC 7%, FW3)",
                "DCLM-Baseline (QC 7%, FW2)",
                "DCLM-Baseline (QC FW 3%)",
                "DCLM-Baseline (QC FW 10%)",
            ],
            "mix": [
                "DCLM-Baseline 25% / Dolma 75%",
                "DCLM-Baseline 50% / Dolma 50%",
                "DCLM-Baseline 75% / Dolma 25%",
            ],
        }
        self.all_data_names = self._all_data_names()
        self.all_param_strs = list(self.model_shapes.keys())
        self.fill_all_model_configs()

    @property
    def model_details_df(self):
        return pd.DataFrame(self.model_configs).T.infer_objects()

    def _all_data_names(self) -> list[str]:
        all_data_names = []
        for ds_list in self.data_recipe_families.values():
            all_data_names.extend(ds_list)
        return all_data_names

    def make_model_config(self, model_size_str: str, **kwargs):
        model_config_base = {
            # Adding other defaults here
            "params": model_size_str,
            "model_size_str": model_size_str,
            "default_seed": 6198,
            "length_str": "5xC",
            "lr_warmup_start": 0.0,
            # Original values
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 12,
            "mlp_ratio": 8,
            "weight_tying": False,
            "alibi": False,
            "rope": True,
            "flash_attention": True,
            "attention_dropout": 0.0,
            "attention_layer_norm": False,
            "include_bias": False,
            "layer_norm_type": "rms",
            "layer_norm_with_affine": True,
            "layer_norm_eps": 1e-6,
            "bias_for_layer_norm": False,
            "attention_layer_norm_with_affine": False,
            "activation_type": "swiglu",
            "residual_dropout": 0.0,
            "embedding_dropout": 0.0,
            "max_sequence_length": 2048,
            "vocab_size": 50280,
            "embedding_size": 50304,
            "eos_token_id": 50279,
            "pad_token_id": 1,
            "init_device": "cuda",
            "init_fn": "normal",
            "init_std": 0.02,
            "init_cutoff_factor": 3,
        }
        mc = {**model_config_base, **kwargs}
        mc["model_size"] = int(self.parse_model_size_str(mc["model_size_str"]))
        mc["batch_size"] = int(self.calc_batch_size(mc["model_size_str"]))
        mc["lr_max"] = self.calc_lr_max(mc["model_size_str"])
        mc["lr_final"] = self.lr_final_ratio * mc["lr_max"]
        mc["warmup_tokens"] = int(self.calc_warmup_tokens(mc["model_size_str"]))
        mc["total_tokens"] = int(
            self.parse_token_length_str(mc["length_str"], mc["model_size_str"])
        )
        mc["lr_decay_tokens"] = int(mc["total_tokens"] - mc["warmup_tokens"])
        mc["total_seqs"] = int(round(mc["total_tokens"] / self.max_seq_len))
        mc["total_steps"] = int(
            math.ceil(mc["total_tokens"] / (mc["batch_size"] * self.max_seq_len))
        )
        mc["warmup_perc"] = mc["warmup_tokens"] / mc["total_tokens"]
        mc["warmup_steps"] = int(
            math.ceil(mc["warmup_tokens"] / (mc["batch_size"] * self.max_seq_len))
        )
        mc["lr_decay_steps"] = int(mc["total_steps"] - mc["warmup_steps"])
        mc["theoretical_tokens_per_step"] = int(
            round(self.max_seq_len * mc["batch_size"])
        )
        mc["10_perc_lrdecay_steps"] = int(round(mc["lr_decay_steps"] * 0.1))
        mc["early_window_10p_end_step"] = (
            mc["warmup_steps"] + mc["10_perc_lrdecay_steps"]
        )
        mc["early_window_perc"] = mc["early_window_10p_end_step"] / mc["total_steps"]
        return mc

    def fill_all_model_configs(self):
        for param_str, cfg in self.model_shapes.items():
            self.model_configs[param_str] = self.make_model_config(param_str, **cfg)

    def parse_model_size_str(self, size_str: str) -> int:
        return self.hardcoded_size_mapping[size_str]

    def parse_token_length_str(self, length_str: str, model_size_str: str) -> int:
        model_size = self.parse_model_size_str(model_size_str)
        length_in_tokens, length_unit = self._number_unit_re.match(
            length_str.strip().upper()
        ).groups()  # type: ignore
        assert length_unit == "XC"
        length_in_tokens = int(length_in_tokens)
        return length_in_tokens * 20 * model_size

    def calc_batch_size(self, model_size_str: str) -> int:
        assert self.max_seq_len == 2_048
        model_size = self.parse_model_size_str(model_size_str)
        batch_size = 160 * (model_size / self.model_size_norm_value) ** self.bs_exponent
        rounding_size = self.gpus_per_node * self.microbatch_size
        batch_size /= rounding_size
        batch_size = round(batch_size)
        batch_size *= rounding_size
        return batch_size

    def calc_lr_max(self, model_size_str: str) -> float:
        model_size = self.parse_model_size_str(model_size_str)
        return (
            self.lr_max_base
            * (model_size / self.model_size_norm_value) ** self.lr_exponent
        )

    def calc_warmup_tokens(self, model_size_str: str) -> int:
        model_size = self.parse_model_size_str(model_size_str)
        bs = self.calc_batch_size(model_size_str)
        # model_size / bs = num_warmup_steps
        # (model_size / bs) * max_seq_len = num_warmup_tokens
        return round(model_size / (bs / self.max_seq_len))


class DataDecide:
    def __init__(self, force_reload=False, verbose=True):
        self.paths = DataDecidePaths()
        self.defaults = DataDecideDefaults()
        self.dwn_drop_cols = ["chinchilla", "tokens", "compute"]
        self.ppl_drop_cols = ["__index_level_0__"]
        self.drop_metrics = [
            "predicted_index_raw",
            "predicted_index_per_token",
            "predicted_index_per_char",
            "predicted_index_per_byte",
            "predicted_index_uncond",
            "logits_per_byte_corr",
        ]
        self.key_cols = ["params", "data", "seed", "step"]
        self.exclude_cols = ["params", "data", "task", "step", "seed"]
        self.ppl_types = self.defaults.ppl_name_map.values()
        self.max_step_to_use = {
            "1B": 67500,
            "750M": 62500,
            "530M": 51250,
            "300M": 45000,
            "150M": 37500,
            "90M": 29901,
            "60M": 29042,
            "20M": 14584,
            "16M": 24432,
            "14M": 21953,
            "10M": 15117,
            "8M": 13039,
            "6M": 9182,
            "4M": 5725,
        }

        # Load from dfs
        self.mmlu_tasks = [
            "mmlu_abstract_algebra",
            "mmlu_anatomy",
            "mmlu_astronomy",
            "mmlu_average",
            "mmlu_business_ethics",
            "mmlu_clinical_knowledge",
            "mmlu_college_biology",
            "mmlu_college_chemistry",
            "mmlu_college_computer_science",
            "mmlu_college_mathematics",
            "mmlu_college_medicine",
            "mmlu_college_physics",
            "mmlu_computer_security",
            "mmlu_conceptual_physics",
            "mmlu_econometrics",
            "mmlu_electrical_engineering",
            "mmlu_elementary_mathematics",
            "mmlu_formal_logic",
            "mmlu_global_facts",
            "mmlu_high_school_biology",
            "mmlu_high_school_chemistry",
            "mmlu_high_school_computer_science",
            "mmlu_high_school_european_history",
            "mmlu_high_school_geography",
            "mmlu_high_school_government_and_politics",
            "mmlu_high_school_macroeconomics",
            "mmlu_high_school_mathematics",
            "mmlu_high_school_microeconomics",
            "mmlu_high_school_physics",
            "mmlu_high_school_psychology",
            "mmlu_high_school_statistics",
            "mmlu_high_school_us_history",
            "mmlu_high_school_world_history",
            "mmlu_human_aging",
            "mmlu_human_sexuality",
            "mmlu_international_law",
            "mmlu_jurisprudence",
            "mmlu_logical_fallacies",
            "mmlu_machine_learning",
            "mmlu_management",
            "mmlu_marketing",
            "mmlu_medical_genetics",
            "mmlu_miscellaneous",
            "mmlu_moral_disputes",
            "mmlu_moral_scenarios",
            "mmlu_nutrition",
            "mmlu_philosophy",
            "mmlu_prehistory",
            "mmlu_professional_accounting",
            "mmlu_professional_law",
            "mmlu_professional_medicine",
            "mmlu_professional_psychology",
            "mmlu_public_relations",
            "mmlu_security_studies",
            "mmlu_sociology",
            "mmlu_us_foreign_policy",
            "mmlu_virology",
            "mmlu_world_religions",
        ]
        self.olmes_tasks = [
            "mmlu_average",
            "arc_challenge",
            "arc_easy",
            "boolq",
            "csqa",
            "hellaswag",
            "openbookqa",
            "piqa",
            "socialiqa",
            "winogrande",
        ]  # just the ones we use
        self.metric_names = [
            "correct_choice",
            "acc_raw",
            "acc_per_token",
            "acc_per_char",
            "acc_per_byte",
            "acc_uncond",
            "no_answer",
            "sum_logits_corr",
            "logits_per_token_corr",
            "logits_per_char_corr",
            "bits_per_byte_corr",
            "correct_prob",
            "correct_prob_per_token",
            "correct_prob_per_char",
            "margin",
            "margin_per_token",
            "margin_per_char",
            "total_prob",
            "total_prob_per_token",
            "total_prob_per_char",
            "uncond_correct_prob",
            "uncond_correct_prob_per_token",
            "uncond_correct_prob_per_char",
            "uncond_total_prob",
            "norm_correct_prob",
            "norm_correct_prob_per_token",
            "norm_correct_prob_per_char",
            "primary_metric",
        ]
        self.data_names = []

        # Prep for df management
        self._setup_dfs = {}
        self._loaded_dfs = {}
        self.setup_all_dfs(force_reload=force_reload, verbose=verbose)
        self._loaded_dfs["ds_details_df"] = self.load_ds_details_df()

    @property
    def all_data_param_combos(self):
        return list(
            itertools.product(
                self.defaults.all_data_names,
                self.defaults.all_param_strs,
            )
        )

    @property
    def ppl_raw_df(self):
        assert "ppl_raw_df" in self._setup_dfs, "ppl_raw_df not setup"
        if "ppl_raw_df" not in self._loaded_dfs:
            self._loaded_dfs["ppl_raw_df"] = pd.read_parquet(
                self._setup_dfs["ppl_raw_df"]
            )
        return self._loaded_dfs["ppl_raw_df"]

    @property
    def dwn_raw_df(self):
        assert "dwn_raw_df" in self._setup_dfs, "dwn_raw_df not setup"
        if "dwn_raw_df" not in self._loaded_dfs:
            self._loaded_dfs["dwn_raw_df"] = pd.read_parquet(
                self._setup_dfs["dwn_raw_df"]
            )
        return self._loaded_dfs["dwn_raw_df"]

    @property
    def ppl_parsed_df(self):
        assert "ppl_parsed_df" in self._setup_dfs, "ppl_parsed_df not setup"
        if "ppl_parsed_df" not in self._loaded_dfs:
            self._loaded_dfs["ppl_parsed_df"] = pd.read_parquet(
                self._setup_dfs["ppl_parsed_df"]
            )
        return self._loaded_dfs["ppl_parsed_df"]

    @property
    def dwn_parsed_df(self):
        assert "dwn_parsed_df" in self._setup_dfs, "dwn_parsed_df not setup"
        if "dwn_parsed_df" not in self._loaded_dfs:
            self._loaded_dfs["dwn_parsed_df"] = pd.read_parquet(
                self._setup_dfs["dwn_parsed_df"]
            )
        return self._loaded_dfs["dwn_parsed_df"]

    @property
    def step_to_token_compute_df(self):
        if "step_to_token_compute_df" in self._setup_dfs:
            self._loaded_dfs["step_to_token_compute_df"] = pd.read_parquet(
                self._setup_dfs["step_to_token_compute_df"]
            )
        return self._loaded_dfs["step_to_token_compute_df"]

    @property
    def full_eval_ds(self):
        assert "full_eval_ds" in self._setup_dfs, "full_eval_ds not setup"
        if "full_eval_ds" not in self._loaded_dfs:
            self._loaded_dfs["full_eval_ds"] = pd.read_parquet(
                self._setup_dfs["full_eval_ds"]
            )
        return self._loaded_dfs["full_eval_ds"]

    @property
    def mean_eval_ds(self):
        assert "mean_eval_ds" in self._setup_dfs, "mean_eval_ds not setup"
        if "mean_eval_ds" not in self._loaded_dfs:
            self._loaded_dfs["mean_eval_ds"] = pd.read_parquet(
                self._setup_dfs["mean_eval_ds"]
            )
        return self._loaded_dfs["mean_eval_ds"]

    @property
    def std_eval_ds(self):
        assert "std_eval_ds" in self._setup_dfs, "std_eval_ds not setup"
        if "std_eval_ds" not in self._loaded_dfs:
            self._loaded_dfs["std_eval_ds"] = pd.read_parquet(
                self._setup_dfs["std_eval_ds"]
            )
        return self._loaded_dfs["std_eval_ds"]

    @property
    def ds_details_df(self):
        return self._loaded_dfs["ds_details_df"]

    @property
    def model_details_df(self):
        return self.defaults.model_details_df

    # ------------ Dataframe Manipulation Helpers ------------

    def filter_by_max_step_to_use(self, df):
        df = df.copy()
        df["max_step_to_use"] = df["params"].map(self.max_step_to_use)
        return df[df["step"] <= df["max_step_to_use"]]

    def merge_in_ds_and_model_details(self, input_df: pd.DataFrame):
        return input_df.merge(
            self.ds_details_df,
            on="data",
            how="left",
        ).merge(
            self.model_details_df,
            on="params",
            how="left",
        )

    def get_max_ppl_vals(self, df: pd.DataFrame):
        ppl_cols = self.ppl_name_map.values()
        return df[ppl_cols].max().reset_index()

    def set_step_val_to_max_ppl_val(self, df: pd.DataFrame, step: int = 0):
        ppl_cols = self.ppl_name_map.values()
        max_ppl_vals = self.get_max_ppl_vals(df)
        df = df.copy()
        step_mask = df["step"] == step
        for col in ppl_cols:
            na_mask = df[col].isna()
            df.loc[step_mask & na_mask, col] = max_ppl_vals[col][0]
        return df

    # ------------ Dataframe Management ------------

    def load_ds_details_df(self):
        df = pd.read_csv(self._setup_dfs["ds_details_df"]).rename(
            columns={
                "dataset": "data",
            }
        )
        df["data"] = (
            df["data"]
            .str.replace("Dolma1.7 (no math code)", "Dolma1.7 (no math, code)")
            .str.replace("DCLM-Baseline (QC 7%", "DCLM-Baseline (QC 7%,")
        )
        return df

    def setup_all_dfs(
        self,
        force_reload=False,
        verbose=False,
    ):
        self._setup_dfs["ds_details_df"] = self.paths.ds_details_path
        # Step 1: Download raw dfs
        if not self.paths.ppl_eval_raw_path.exists() or force_reload:
            if verbose:
                print("Downloading raw dfs")
            download_dataset(
                path=self.paths.ppl_eval_raw_path,
                repo_id=self.defaults.hf_ds_names["perplexity_eval_ds"],
                split="train",
                force_reload=force_reload,
            )
        self._setup_dfs["ppl_raw_df"] = self.paths.ppl_eval_raw_path
        if not self.paths.downstream_eval_raw_path.exists() or force_reload:
            download_dataset(
                path=self.paths.downstream_eval_raw_path,
                repo_id=self.defaults.hf_ds_names["downstream_eval_ds"],
                split="train",
                force_reload=force_reload,
            )
        self._setup_dfs["dwn_raw_df"] = self.paths.downstream_eval_raw_path

        # Step 2: Extract per-param step-to-token and step-to-compute mapping
        if not self.paths.step_to_token_compute_path.exists() or force_reload:
            if verbose:
                print("Extracting step-to-token and step-to-compute mapping")
            dwn_df = pd.read_parquet(self._setup_dfs["dwn_raw_df"])
            step_to_token_compute_df = make_step_to_token_compute_df(dwn_df)
            step_to_token_compute_df.to_parquet(self.paths.step_to_token_compute_path)
        self._setup_dfs["step_to_token_compute_df"] = (
            self.paths.step_to_token_compute_path
        )

        # Step 3: Parse eval dfs
        if not self.paths.ppl_eval_parsed_path.exists() or force_reload:
            ppl_df = pd.read_parquet(self._setup_dfs["ppl_raw_df"])
            ppl_parsed_df = self.parse_ppl_df(ppl_df)
            ppl_parsed_df.to_parquet(self.paths.ppl_eval_parsed_path)
        self._setup_dfs["ppl_parsed_df"] = self.paths.ppl_eval_parsed_path
        if not self.paths.downstream_eval_parsed_path.exists() or force_reload:
            if verbose:
                print("Parsing eval dfs, this may take a while...")
            dwn_df = pd.read_parquet(self._setup_dfs["dwn_raw_df"])
            dwn_parsed_df = self.parse_dwn_df(dwn_df)
            dwn_parsed_df.to_parquet(self.paths.downstream_eval_parsed_path)
        self._setup_dfs["dwn_parsed_df"] = self.paths.downstream_eval_parsed_path

        # Step 4: Create full eval df
        if not self.paths.full_eval_ds_path.exists() or force_reload:
            full_eval_ds = self.create_full_eval_df(
                self.dwn_parsed_df,
                self.ppl_parsed_df,
                self.step_to_token_compute_df,
            )
            full_eval_ds.to_parquet(self.paths.full_eval_ds_path)
        self._setup_dfs["full_eval_ds"] = self.paths.full_eval_ds_path

        # Step 5: create mean and std eval dfs
        if (
            not self.paths.mean_eval_ds_path.exists()
            or not self.paths.std_eval_ds_path.exists()
            or force_reload
        ):
            mean_eval_ds, std_eval_ds = self.create_mean_std_df(self.full_eval_ds)
            mean_eval_ds.to_parquet(self.paths.mean_eval_ds_path)
            std_eval_ds.to_parquet(self.paths.std_eval_ds_path)
        self._setup_dfs["mean_eval_ds"] = self.paths.mean_eval_ds_path
        self._setup_dfs["std_eval_ds"] = self.paths.std_eval_ds_path
        self._setup_dfs["model_details_df"] = None
        self._loaded_dfs = {
            "model_details_df": self.defaults.model_details_df,
        }
        print(">> Finished setting up DataDecide dataframes.")

    def load_df(self, df_name: str) -> None:
        assert df_name in self._setup_dfs, f"df_name {df_name} not setup"
        if df_name not in self._loaded_dfs:
            self._loaded_dfs[df_name] = pd.read_parquet(self._setup_dfs[df_name])

    def index_dfs(self, df_name, params, data, step):
        if df_name not in self._loaded_dfs:
            self._loaded_dfs[df_name] = pd.read_parquet(self._setup_dfs[df_name])
        df = self._loaded_dfs[df_name]
        return df[
            (df["params"] == params) & (df["data"] == data) & (df["step"] == step)
        ]

    # ------------ Parsing Helpers ------------

    def parse_ppl_df(self, ppl_df: pd.DataFrame) -> pd.DataFrame:
        df = ppl_df.copy()
        df = df.drop(columns=self.ppl_drop_cols)
        df = df.rename(columns=self.defaults.ppl_name_map)
        df = reorder_df_cols(df, self.key_cols)
        df["seed"] = df["seed"].map(self.defaults.seed_map)
        return df

    def parse_dwn_df(self, dwn_df: pd.DataFrame) -> pd.DataFrame:
        df = dwn_df.copy()
        df = df.drop(columns=self.dwn_drop_cols)
        df = list_col_to_columns(df, "metrics")
        df = df.drop(columns=self.drop_metrics)
        df = self.average_mmlu_metrics(df)
        df = self.pivot_task_metrics_rows_to_cols(df)
        df = reorder_df_cols(df, self.key_cols)
        df["seed"] = df["seed"].map(self.defaults.seed_map)
        return df

    def create_full_eval_df(
        self,
        dwn_parsed_df: pd.DataFrame,
        ppl_parsed_df: pd.DataFrame,
        step_to_token_compute_df: pd.DataFrame,
    ) -> pd.DataFrame:
        merged_df = dwn_parsed_df.merge(
            ppl_parsed_df,
            on=["params", "data", "seed", "step"],
            how="outer",
            suffixes=("_dwn", "_ppl"),
        )
        merged_df = (
            merged_df.merge(step_to_token_compute_df, on="params", how="left")
            .assign(
                tokens=lambda x: x["step"] * x["tokens_per_step"],
                compute=lambda x: x["step"] * x["compute_per_step"],
            )
            .drop(columns=["tokens_per_step", "compute_per_step"])
        )
        merged_df = reorder_df_cols(merged_df, self.key_cols + ["tokens", "compute"])
        return merged_df

    def create_mean_std_df(
        self, merged_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        group_cols_no_seed = [c for c in self.key_cols if c != "seed"]
        mean_df = (
            merged_df.drop(columns=["seed"])
            .groupby(group_cols_no_seed)
            .mean(numeric_only=True)
            .reset_index()
        )
        std_df = (
            merged_df.drop(columns=["seed"])
            .groupby(group_cols_no_seed)
            .std(numeric_only=True)
            .reset_index()
        )
        return mean_df, std_df

    def average_mmlu_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        self.mmlu_tasks = [
            task for task in df["task"].unique() if "mmlu" in task.lower()
        ]
        mmlu_df = df[df["task"].isin(self.mmlu_tasks)].drop(columns=["task"])
        if len(self.metric_names) == 0:
            self.metric_names = [
                col
                for col in df.columns
                if col not in self.exclude_cols and col not in self.drop_metrics
            ]
        mmlu_avg = mmlu_df.groupby(self.key_cols).agg("mean").reset_index()
        mmlu_avg["task"] = "mmlu_average"
        return pd.concat([df, mmlu_avg], ignore_index=True)

    def pivot_task_metrics_rows_to_cols(self, dwn_df: pd.DataFrame) -> pd.DataFrame:
        pivoted_metrics = []
        for metric_col in self.metric_names:
            pivoted = dwn_df.pivot_table(
                index=self.key_cols,
                columns="task",
                values=metric_col,
                aggfunc="first",
            )
            pivoted.columns = [f"{task}_{metric_col}" for task in pivoted.columns]
            pivoted_metrics.append(pivoted)
        new_dwn_df = pd.concat(pivoted_metrics, axis=1).reset_index()
        return new_dwn_df


# ------------ Parsing Functions ------------
def make_step_to_token_compute_df(dwn_df: pd.DataFrame) -> pd.DataFrame:
    assert all(
        [col in dwn_df.columns for col in ["params", "step", "tokens", "compute"]]
    )
    step_map = dwn_df[dwn_df["step"] > 0].copy()
    step_map["tokens_per_step"] = step_map["tokens"] / step_map["step"]
    step_map["compute_per_step"] = step_map["compute"] / step_map["step"]
    return step_map[["params", "tokens_per_step", "compute_per_step"]].drop_duplicates()


def list_col_to_columns(orig_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    json_data = orig_df[col_name].str.replace("'", '"')  # Single to double quotes
    df = pd.json_normalize(json_data.apply(json.loads))
    df = pd.concat([orig_df.drop(col_name, axis=1), df], axis=1)
    # df = pd.json_normalize(orig_df[col_name].apply(ast.literal_eval))
    # df = pd.concat([orig_df.drop(col_name, axis=1), df], axis=1)
    return df


def reorder_df_cols(df: pd.DataFrame, prefix_order: list[str]) -> pd.DataFrame:
    df = df.copy()
    return df[prefix_order + [col for col in df.columns if col not in prefix_order]]
