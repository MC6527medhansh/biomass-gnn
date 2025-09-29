from dataclasses import dataclass
from typing import Dict, List
import os, csv, glob

@dataclass
class MediaProfile:
    name: str
    bounds: Dict[str, float]  # exchange_rxn_id -> lower_bound (negative means import allowed)

def load_media_dir(media_dir: str) -> List[MediaProfile]:
    """Load all *.csv media files; each line: EX_rxn_id,lower_bound"""
    profiles: List[MediaProfile] = []
    for path in sorted(glob.glob(os.path.join(media_dir, '*.csv'))):
        name = os.path.splitext(os.path.basename(path))[0]
        bounds = {}
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                rxn_id, lower_bound = row[0].strip(), float(row[1])
                bounds[rxn_id] = lower_bound
        profiles.append(MediaProfile(name=name, bounds=bounds))
    return profiles
