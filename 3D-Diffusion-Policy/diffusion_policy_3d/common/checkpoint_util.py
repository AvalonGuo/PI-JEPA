import json
import os
import time
from typing import Optional, Dict


class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        del self.path_value_map[delete_path]
        self.path_value_map[ckpt_path] = value

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if os.path.exists(delete_path) and delete_path != ckpt_path:
            os.remove(delete_path)
        return ckpt_path


def best_record_path(save_dir):
    return os.path.join(save_dir, 'best.json')


def load_best_record(save_dir):
    path = best_record_path(save_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (ValueError, OSError):
        return None
    return record if isinstance(record, dict) else None


def write_best_record(save_dir, filename, monitor_key, mode, score,
                      metrics=None, epoch=None, global_step=None):
    os.makedirs(save_dir, exist_ok=True)
    record = {
        'monitor': monitor_key,
        'mode': mode,
        'score': float(score),
        'epoch': None if epoch is None else int(epoch),
        'global_step': None if global_step is None else int(global_step),
        'file': filename,
        'alias': 'best.ckpt',
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': dict(metrics or {}),
    }
    path = best_record_path(save_dir)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as handle:
        json.dump(record, handle, indent=2)
    os.replace(tmp_path, path)
    return path
