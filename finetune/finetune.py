import os 
import uuid
import sqlite3
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from transformers.hf_argparser import HfArgumentParser
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import is_torch_bf16_gpu_available

from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_datasets,
    load_rl_datasets,
    print_axolotl_text_art,
)
from axolotl.utils.config import (
    normalize_cfg_datasets,
    normalize_config,
    validate_config,
)
from axolotl.utils.mlflow_ import setup_mlflow_env_vars
from axolotl.utils.trainer import prepare_optim_env
from axolotl.utils.wandb_ import setup_wandb_env_vars
from axolotl.utils.dict import DictDefault
from axolotl.common.cli import TrainerCliArgs
from axolotl.prompt_strategies.sharegpt import register_chatml_template
from axolotl.train import train

LOG = logging.getLogger("altus.finetune.finetune")

def load_cfg(config: Union[str, Path, Dict], **kwargs):
    if isinstance(config, Dict):
        # load the config dict
        cfg: DictDefault = DictDefault(config)
    elif isinstance(config, Path) or isinstance(config, str):
        # load the config from the yaml file
        with open(config, encoding="utf-8") as file:
            cfg: DictDefault = DictDefault(yaml.safe_load(file))
    else:
        NotImplementedError
    
    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    cfg_keys = cfg.keys()
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys or not cfg.strict:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]

    if isinstance(config, Dict):
        cfg.axolotl_config_path = ""
    else:
        cfg.axolotl_config_path = config

    try:
        device_props = torch.cuda.get_device_properties("cuda")
        gpu_version = "sm_" + str(device_props.major) + str(device_props.minor)
    except:  # pylint: disable=bare-except # noqa: E722
        gpu_version = None

    cfg = validate_config(
        cfg,
        capabilities={
            "bf16": is_torch_bf16_gpu_available(),
            "n_gpu": os.environ.get("WORLD_SIZE", 1),
            "compute_capability": gpu_version,
        },
    )

    prepare_optim_env(cfg)

    normalize_config(cfg)

    normalize_cfg_datasets(cfg)

    setup_wandb_env_vars(cfg)

    setup_mlflow_env_vars(cfg)

    return cfg
    
class FinetuneConfig:
    def __init__(self, config: Union[Path, str, Dict]):
        self.cfg = load_cfg(config)

class Finetune:
    def __init__(self, config: FinetuneConfig, **kwargs):
        self.config = config.cfg
        self.meta = {"job_id": None, "completed": False}

        # saving meta, configs
        self.db_path = 'training_jobs.db'
        self.initialize_db()

        # Optional parameters with their default values
        self.cli_args = TrainerCliArgs
        self.cli_args.debug = kwargs.get('debug', False)
        self.cli_args.debug_text_only = kwargs.get('debug_text_only', False)
        self.cli_args.debug_num_examples = kwargs.get('debug_num_examples', 5)
        self.cli_args.inference = kwargs.get('inference', False)
        self.cli_args.merge_lora = kwargs.get('merge_lora', False)
        self.cli_args.prompter = kwargs.get('prompter', None)
        self.cli_args.shard = kwargs.get('shard', False)

    def initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_status (
                job_id TEXT PRIMARY KEY,
                completed BOOLEAN NOT NULL,
                config TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_status_to_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Serialize config with json
        cursor.execute("INSERT OR REPLACE INTO training_status (job_id, completed, config) VALUES (?, ?, ?)",
                       (self.meta["job_id"], self.meta["completed"], str(self.config)))
        conn.commit()
        conn.close()

    def get_status_from_db(self, job_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT completed, config FROM training_status WHERE job_id = ?", (job_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            config = result[1]  # load the configuration
            completed_status = "Training completed" if result[0] else "Training in progress"
            return completed_status, config  # Optionally return the config as well
        return "No job found with the given ID", None
        
    def run(self) -> str:
        check_accelerate_default_config()
        check_user_token()

        self.meta["job_id"] = str(uuid.uuid1())  # Unique identifier
        print(f'[INFO] Job ID: {self.meta["job_id"]}')

        if self.config.get('chat_template') == "chatml" and self.config.get('default_system_message'):
            LOG.info(f"ChatML set. Adding default system message: {self.config['default_system_message']}")
            register_chatml_template(self.config['default_system_message'])
        else:
            register_chatml_template()

        if self.config.get('rl') and self.config['rl'] != "orpo":
            dataset_meta = load_rl_datasets(cfg=self.config, cli_args=self.cli_args)
        else:
            dataset_meta = load_datasets(cfg=self.config, cli_args=self.cli_args)

        self.log_status_to_db() # log to db (before training)

        _ = train(cfg=self.config, cli_args=self.cli_args, dataset_meta=dataset_meta)
        self.meta["completed"] = True

        self.log_status_to_db() # log to db (after training)
        return 
        
    def status(self, job_id: str):
        status, config = self.get_status_from_db(job_id)
        return config, status