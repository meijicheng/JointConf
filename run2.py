import asyncio
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
    labels = []
    for i in result:
        labels.append(len(i))
    result_label = max(labels)
    for i in result:
        if len(i) == result_label:
            return np.mean(i)


def translate_config_to_numeric(default_config, app_setting):
    config = dict(app_setting)

    # default configs, need to transform category into values
    default_config_v = {}
    for k, v in default_config.items():
        v_range = config[k].get('range')
        if v_range:
            # v_range = str(v_range).lower()
            default_config_v[k] = v_range.index(v)
        else:
            default_config_v[k] = v

    return default_config_v


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
    valid_num = 0
    # 读取上一段运行的结果
    for i in range(init_id + 1):
        exist_process_engine_config_path = result_dir / f'{i}_process_engine_config.yml'
        exist_storage_engine_config_path = result_dir / f'{i}_storage_engine_config.yml'
        # 读取配置
        exist_process_engine_config = yaml.load(
            exist_process_engine_config_path.read_text()
        )
        exist_process_engine_config_numeric = translate_config_to_numeric(exist_process_engine_config,
                                                                           process_engine_setting)
        exist_storage_engine_config = yaml.load(
            exist_storage_engine_config_path.read_text()
        )
        exist_storage_engine_config_numeric = translate_config_to_numeric(exist_storage_engine_config,
                                                                           storage_engine_setting)

        exist_config_numeric = {
            **(exist_process_engine_config_numeric if test_config.hosts.processEngine.tune else {}),
            **(exist_storage_engine_config_numeric if test_config.hosts.storageEngine.tune else {}),
        }

        exist_results = []
        # 读取结果
        for j in range(test_config.optimizer.repitition):
            result = parse_result(
                tester_name=test_config.tester,
                result_dir=result_dir,
                task_id=i,
                rep=j,
                printer=_print
            )
            if result is not None:
                exist_results.append(result)
            else:
                break
        if len(exist_results) < test_config.optimizer.repitition:
            print("skip")
            continue

        print(exist_results)
        exist_result = - get_Kmeans_result(exist_results) if len(exist_results) > 0 else .0
        print(exist_result)

        if exist_result > -1500.0:
            print(exist_result)
            print("skip")
            continue

        # 加入观察集
        if i != 0:  # not adding default info, 'cause default cannot convert to numeric form
            # get_Kmeans_result
            optimizer.add_observation(
                (exist_config_numeric, exist_result)
            )
            valid_num += 1

            if hasattr(optimizer, 'dump_state'):
                optimizer.dump_state(result_dir / f'{i}_optimizer_state')
    if valid_num >= test_config.optimizer.iter_limit:
        return
    if hasattr(optimizer, 'set_status_file'):
        optimizer.set_status_file(result_dir / 'optimizer_status')
    skip_flag = False
    task_id = init_id
    processEngine, storageEngine = test_config.hosts.processEngine, test_config.hosts.storageEngine
    while valid_num < test_config.optimizer.iter_limit:
        task_id += 1
        # if task_id != 0 and task_id % test_config.optimizer.reboot_interval == 0:
        #   _print('rebooting...')

        if task_id == 0:  # use default config
            sampled_config_numeric, sampled_config = None, get_default({
                **process_engine_setting,
                **storage_engine_setting
            })
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

        # init process engine and storage engine
        if task_id == 0:
            # restore environment
            await restore_environment(
                task_id=task_id,
                processEngine=processEngine,
                storageEngine=storageEngine
            )

        result = await init_environment(
            task_name=test_config.task_name,
            task_id=task_id,
            processEngine=processEngine,
            storageEngine=storageEngine,
        )

        if not result:
            # close cassandra and delete all files
            skip_flag = True

        metric_results = []
        if not skip_flag:
            # update configs
            await update_configs(
                task_name=test_config.task_name,
                task_id=task_id,
                processEngine=processEngine,
                storageEngine=storageEngine,
            )

            for rep in range(test_config.optimizer.repitition):
                run_result = await run_workload(
                    task_name=test_config.task_name,
                    task_id=task_id,
                    rep=rep,
                    processEngine=processEngine,
                    storageEngine=storageEngine,
                    workload=test_config.workload
                )

                # after test, collect metrics for evaluation
                _print(f'{task_id} - {rep}: parsing result...')
                result = parse_result(
                    tester_name=test_config.tester,
                    result_dir=result_dir,
                    task_id=task_id,
                    rep=rep,
                    printer=_print
                )
                if result is not None and run_result:
                    metric_results.append(result)
                    print(result)
                else:
                    skip_flag = True
                    _print(f'{task_id} - {rep}: parsing result failed.')
                    break
                _print(f'{task_id} - {rep}: done.')

        if skip_flag:
            _print(f'{task_id} : error, skip .')
            await run_playbook(
                restore_storage_engine_cassandra_playbook_path,
                host=storageEngine.host,
                user=storageEngine.user,
                task_name=test_config.task_name,
                task_id=task_id,
            )
            _print(f'{task_id} : skip done.')
            skip_flag = False
            continue

        print(metric_results)
        metric_result = - get_Kmeans_result(metric_results) if len(metric_results) > 0 else .0
        print(metric_result)

        if metric_result > -1500.0:
            print(metric_result)
            print("skip")
            continue

        # after
        if task_id != 0:  # not adding default info, 'cause default cannot convert to numeric form
            # get_Kmeans_result
            optimizer.add_observation(
                (sampled_config_numeric, metric_result)
            )
            valid_num += 1
            if hasattr(optimizer, 'dump_state'):
                optimizer.dump_state(result_dir / f'{task_id}_optimizer_state')

    # after reaching iter limit
    global proj_root

    # # clean up
    # _print('experiment finished, cleaning up and reboot...')
    # await run_playbook(
    #     playbook_path=db_dir / 'playbook/reboot.yml',
    #     task_name=test_config.task_name,
    #     db_name=test_config.target,
    #     host=test_config.hosts.tester,
    #     user=test_config.users.janusgraph
    # )
    #
    # await run_playbook(
    #     playbook_path=db_dir / 'playbook/reboot.yml',
    #     task_name=test_config.task_name,
    #     db_name=test_config.target,
    #     host=test_config.hosts.testee,
    #     user=test_config.users.hbase,
    # )


async def init_environment(task_name, task_id, processEngine, storageEngine):
    global storage_engine_init_playbook_path
    global process_engine_init_playbook_path

    result = True

    try:
        _print(f'{task_id} :init environment...')

        _print(f'{task_id} :init storage engine environment...')
        await run_playbook(
            storage_engine_init_playbook_path,
            host=storageEngine.host,
            user=storageEngine.user,
            ip=storageEngine.ip,
            task_name=task_name,
            task_id=task_id,
        )
        _print(f'{task_id}:init storage engine done...')

        _print(f'{task_id}: init process engine environment...')
        await run_playbook(
            process_engine_init_playbook_path,
            host=processEngine.host,
            user=processEngine.user,
            name=storageEngine.name,
            ip=storageEngine.ip,
            task_name=task_name,
            task_id=task_id,
        )
        _print(f'{task_id}:init process engine done...')

        _print(f'{task_id}:init environment done...')
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_init_environment_error.log'
        errlog_path.write_text(str(e))
        print(e)
        result = False
    return result


async def update_configs(task_name, task_id, processEngine, storageEngine):
    global storage_engine_update_config_playbook_path
    global process_engine_update_config_playbook_path

    try:
        _print(f'{task_id} : update configs...')

        _print(f'{task_id} : update storage engine config...')
        await run_playbook(
            storage_engine_update_config_playbook_path,
            host=storageEngine.host,
            user=storageEngine.user,
            ip=storageEngine.ip,
            task_name=task_name,
            task_id=task_id,
        )
        _print(f'{task_id}: update storage engine config done...')

        _print(f'{task_id}: update process engine config...')
        await run_playbook(
            process_engine_update_config_playbook_path,
            host=processEngine.host,
            user=processEngine.user,
            name=storageEngine.name,
            ip=storageEngine.ip,
            task_name=task_name,
            task_id=task_id,
        )
        _print(f'{task_id}: update process engine config done...')
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_update_configs_error.log'
        errlog_path.write_text(str(e))
        print(e)


async def run_workload(task_name, task_id, rep, processEngine, storageEngine, workload):
    global run_workload_playbook_path
    result = False
    _print(f'{task_id}: carrying out #{rep} repetition run workload...')

    try:
        _print(f'{task_id} - {rep}: run workload({workload}) ')
        await run_playbook(
            run_workload_playbook_path,
            processEngine=processEngine.__dict__,
            storageEngine=storageEngine.__dict__,
            task_name=task_name,
            task_id=task_id,
            workload=workload,
            rep=rep
        )
        _print(f'{task_id} - {rep}: run workload({workload}) done...')
        return not result
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_run_workload_{rep}.log'
        errlog_path.write_text(str(e))
        print(e)
        return result


async def restore_environment(task_id, processEngine, storageEngine):
    global restore_environment_playbook_path

    try:
        _print(f'{task_id}: restore environment...')
        _print(f'{task_id}: restore process engine environment...')
        await run_playbook(
            restore_environment_playbook_path,
            host=processEngine.host,
            user=processEngine.user,
        )
        _print(f'{task_id}: restore process engine environment done...')
        _print(f'{task_id}: restore storage engine environment...')
        await run_playbook(
            restore_environment_playbook_path,
            host=storageEngine.host,
            user=storageEngine.user,
        )
        _print(f'{task_id}: restore storage engine environment done...')
        _print(f'{task_id}: restore environment done...')
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_restore_environment_error.log'
        errlog_path.write_text(str(e))
        print(e)


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

restore_storage_engine_cassandra_playbook_path = db_dir / 'playbook/restore_storage_engine_cassandra.yml'

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
