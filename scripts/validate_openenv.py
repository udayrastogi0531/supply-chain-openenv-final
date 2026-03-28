import os
import yaml


def validate_openenv_manifest(path='openenv.yaml'):
    if not os.path.exists(path):
        raise FileNotFoundError('openenv.yaml is missing')
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    assert 'openenv' in data, 'openenv key missing'
    env = data['openenv']
    assert env.get('name') is not None, 'name missing'
    assert isinstance(env.get('tasks'), list) and len(env.get('tasks')) >= 3, 'need 3 tasks'
    assert 'action_space' in env
    assert 'observation_space' in env


if __name__ == '__main__':
    try:
        validate_openenv_manifest()
        print('openenv.yaml valid')
    except Exception as e:
        print('openenv validation failed:', e)
        raise
