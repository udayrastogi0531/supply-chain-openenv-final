import os
import json
from datetime import datetime


def record_trajectory(task, trajectory, root_dir='logs'):
    os.makedirs(root_dir, exist_ok=True)
    filename = f"{task}_trajectory_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(root_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(trajectory, f, indent=2)
    return path


if __name__ == '__main__':
    print('This script provides trajectory saving helper')
