from project.utils.DataLoader import RecommendationDataset
from project.utils.config_utils import file_loader
import pandas as pd

class ItemDataset(RecommendationDataset):
    def __init__(self, config_path, pkl_path, max_seq_len=3):
        """

        :param config_path: file path of config
        :param tower_name: tower name
        :param pkl_path: file path of pkl
        :param max_seq_len: max length of sequence
        """
        raw_cfg = file_loader(config_path)
        cfg = raw_cfg.get("two_tower", {}).get("item_tower", {})
        self.cfg = cfg
        if len(cfg) == 0:
            raise RuntimeError("Config loading failed, no items in item tower config, do you need to run config_utils.py to dump sample config?")
        self.data = pd.read_pickle(pkl_path)
        self.max_seq_len = max_seq_len
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return self._get_tower_meta(row, self.cfg, self.max_seq_len)
