import asyncio
import math

import yaml
import re
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from statistics import mean
from sklearn.cluster import KMeans
from lib.other import parse_cmd, run_playbook, get_default
from lib.optimizer import create_optimizer
from lib.result_parser import parse_result


def translate_config_to_numeric(sample_config, app_setting):
    config = dict(app_setting)
    # default configs, need to transform category into values
    sample_config_v = {}
    for k, v in sample_config.items():
        v_range = config[k].get('range')
        if v_range:
            sample_config_v[k] = v_range.index(v)
        else:
            sample_config_v[k] = v
    return sample_config_v

def get_Kmeans_result(data, clusters=2):
    data_ = []
    for i in data:
        data_.append([i])
    data = data_
    estimator = KMeans(n_clusters=clusters)
    y = estimator.fit_predict(data)
    result = []
    for i in range(clusters):
        cluster = []
        for j in range(len(data)):
            if y[j] == i:
                cluster.append(data[j][0])
        result.append(cluster)
    print(result)
    labels = []
    for i in result:
        labels.append(len(i))
    result_label = max(labels)
    for i in result:
        if len(i) == result_label:
            return np.mean(i)


def find_exist_task_result():
    task_id = -1
    regexp = re.compile(r'(\d+)_.+')
    if result_dir.exists():
        for p in result_dir.iterdir():
            if p.is_file():
                res = regexp.match(p.name)
                if res:
                    task_id = max(task_id, int(res.group(1)))
    return None if task_id == -1 else task_id


def divide_config(sampled_config, process_engine_setting, storage_engine_setting):
    for k in sampled_config.keys():
        if type(sampled_config[k]) is bool:
            # make sure no uppercase 'True/False' literal in result
            sampled_config[k] = str(sampled_config[k]).lower()
        elif type(sampled_config[k]) is np.float64:
            sampled_config[k] = float(sampled_config[k])

    sampled_process_engine_config = dict(
        ((k, v) for k, v in sampled_config.items() if k in process_engine_setting)
    )
    sampled_storage_engine_config = dict(
        ((k, v) for k, v in sampled_config.items() if k in storage_engine_setting)
    )
    return sampled_process_engine_config, sampled_storage_engine_config


def _print(msg):
    print(f'[{datetime.now()}] {test_config.task_name} - {msg}')
    # print('[' + datetime.now() + ']')


async def main(test_config, init_id, process_engine_setting, storage_engine_setting):
    # create optimizer
    optimizer = create_optimizer(
        test_config.optimizer.name,
        {
            **(process_engine_setting if test_config.hosts.processEngine.tune else {}),
            **(storage_engine_setting if test_config.hosts.storageEngine.tune else {}),
        },
        extra_vars=test_config.optimizer.extra_vars
    )
    if hasattr(optimizer, 'set_status_file'):
        optimizer.set_status_file(result_dir / 'optimizer_status')
    task_id = init_id
    processEngine, storageEngine = test_config.hosts.processEngine, test_config.hosts.storageEngine
    while task_id < test_config.optimizer.iter_limit:
        task_id += 1
        # if task_id != 0 and task_id % test_config.optimizer.reboot_interval == 0:
        #   _print('rebooting...')

        if task_id == 0:  # use default config
            sampled_config_numeric, sampled_config = None, get_default({
                **process_engine_setting,
                **storage_engine_setting
            })
            all_config_setting = {
                **process_engine_setting,
                **storage_engine_setting
            }
            sampled_config_numeric = translate_config_to_numeric(sampled_config, all_config_setting)
        else:
            try:
                sampled_config_numeric, sampled_config = optimizer.get_conf()
            except StopIteration:
                # all configuration emitted
                return

        print(sampled_config)
        # - divide sampled config process engine and storage engine
        sampled_process_engine_config, sampled_storage_engine_config = divide_config(
            sampled_config,
            process_engine_setting=process_engine_setting,
            storage_engine_setting=storage_engine_setting
        )
        # if tune_app is off, just give sample_app_config a default value
        if test_config.hosts.processEngine.tune is False:
            sampled_process_engine_config = get_default(process_engine_setting)
        if test_config.hosts.storageEngine.tune is False:
            sampled_storage_engine_config = get_default(storage_engine_setting)

        # - dump configs
        process_engine_config_path = result_dir / f'{task_id}_process_engine_config.yml'
        process_engine_config_path.write_text(
            yaml.dump(sampled_process_engine_config, default_flow_style=False)
        )
        storage_engine_config_path = result_dir / f'{task_id}_storage_engine_config.yml'
        storage_engine_config_path.write_text(
            yaml.dump(sampled_storage_engine_config, default_flow_style=False)
        )
        _print(f'{task_id}:  process_engine_config & storage_engine_config generated.')

        metric_results = []
        qua_sum = 0
        for k, v in sampled_config_numeric.items():
            qua_sum += int(v) ** 2
        result = - (math.cos(qua_sum ** 1 / 2 * 12) + 1) / (0.5 * qua_sum + 2)

        print(result)
        metric_results.append(result)

        # after
        if task_id != 0:  # not adding default info, 'cause default cannot convert to numeric form
            metric_result = - mean(metric_results) if len(metric_results) > 0 else .0
            # get_Kmeans_result
            optimizer.add_observation(
                ((sampled_config, sampled_config_numeric), metric_result)
            )
            if hasattr(optimizer, 'dump_state'):
                optimizer.dump_state(result_dir / f'{task_id}_optimizer_state')

    # after reaching iter limit
    global proj_root


# -------------------------------------------------------------------------------------------------------
test_config = parse_cmd()
assert test_config is not None

proj_root = Path(__file__, '../../..').resolve()

db_dir = proj_root / f'target/{test_config.target}'
result_dir = db_dir / f'results/{test_config.task_name}'

storage_engine_init_playbook_path = db_dir / f'playbook/storage_engine_{test_config.hosts.storageEngine.name}_init.yml'
process_engine_init_playbook_path = db_dir / f'playbook/process_engine_{test_config.hosts.processEngine.name}_init.yml'

storage_engine_update_config_playbook_path = db_dir / f'playbook/storage_engine_{test_config.hosts.storageEngine.name}' \
                                                      f'_update_config.yml '
process_engine_update_config_playbook_path = db_dir / f'playbook/process_engine_{test_config.hosts.processEngine.name}' \
                                                      f'_update_config.yml'

run_workload_playbook_path = db_dir / 'playbook/tester.yml'

restore_environment_playbook_path = db_dir / 'playbook/restore_environment.yml'

workload_path = db_dir / f'workload/{test_config.workload}'
# Hdconfigor/target/hbase/os_configs_info.yml
storage_engine_setting_path = proj_root / \
                              f'target/{test_config.target}/storage_engine_' \
                              f'{test_config.hosts.storageEngine.name}_configs_info.yml'
# Hdconfigor/target/hbase/app_configs_info.yml
process_engine_setting_path = proj_root / \
                              f'target/{test_config.target}/process_engine_' \
                              f'{test_config.hosts.processEngine.name}_configs_info.yml'
process_storage_engine_setting_path = proj_root / \
                              f'target/{test_config.target}/process_engine_' \
                              f'{test_config.hosts.processEngine.name}_{test_config.hosts.storageEngine.name}_' \
                              f'configs_info.yml'


init_id = -1

# check existing results, find minimum available task_id
exist_task_id = find_exist_task_result()

if exist_task_id is not None:
    _print(f'previous results found, with max task_id={exist_task_id}')
    policy = test_config.exist
    if policy == 'delete':
        for file in sorted(result_dir.glob('*')):
            file.unlink()
        _print('all deleted')
    elif policy == 'continue':
        _print(f'continue with task_id={exist_task_id + 1}')
        init_id = exist_task_id
    else:
        _print('set \'exist\' to \'delete\' or \'continue\' to specify what to do, exiting...')
        sys.exit(0)

# create dirs
result_dir.mkdir(parents=True, exist_ok=True)

# dump test configs
(result_dir / 'test_config.yml').write_text(
    yaml.dump(test_config, default_flow_style=False)
)
_print('test_config.yml dumped')

# read parameters for tuning
storage_engine_setting = yaml.load(storage_engine_setting_path.read_text())  # pylint: disable=E1101
process_engine_setting = yaml.load(process_engine_setting_path.read_text())  # pylint: disable=E1101
process_storage_engine_setting = yaml.load(process_storage_engine_setting_path.read_text())  # pylint: disable=E1101
process_engine_setting = {
    **process_engine_setting,
    **process_storage_engine_setting
}

# event loop, main() is async
loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(
        test_config=test_config,
        init_id=init_id,
        process_engine_setting=process_engine_setting,
        storage_engine_setting=storage_engine_setting
    )
)
loop.close()
