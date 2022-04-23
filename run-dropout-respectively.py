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
    print(result)
    for i in result:
        if len(i) == result_label:
            return np.mean(i)

def get_best_config(test_config):
    collect_results = []
    for i in range(test_config.optimizer.iter_limit + 1):
        results = []
        for j in range(test_config.optimizer.repitition):
            result = parse_result(
                tester_name=test_config.tester,
                result_dir=result_dir,
                task_id=i,
                rep=j,
                printer=_print
            )
            results.append(result)
        collect_results.append(get_Kmeans_result(results))
    index = collect_results.index(min(collect_results))
    best_config = result_dir / f'{index}_app_config.yml'
    app_setting = yaml.load(best_config.read_text())
    return app_setting


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


def divide_config(sampled_config, os_setting, app_setting):
    for k in sampled_config.keys():
        if type(sampled_config[k]) is bool:
            # make sure no uppercase 'True/False' literal in result
            sampled_config[k] = str(sampled_config[k]).lower()
        elif type(sampled_config[k]) is np.float64:
            sampled_config[k] = float(sampled_config[k])

    sampled_os_config = dict(
        ((k, v) for k, v in sampled_config.items() if k in os_setting)
    )
    sampled_app_config = dict(
        ((k, v) for k, v in sampled_config.items() if k in app_setting)
    )
    return sampled_os_config, sampled_app_config


def _print(msg):
    print(f'[{datetime.now()}] {test_config.task_name} - {msg}')
    # print('[' + datetime.now() + ']')


def translate_config_to_numeric(default_config,app_setting):
  print('show default_config:')
  print(default_config)

  config=dict(app_setting)

  #default configs, need to transform category into values
  default_config_v={}
  for k, v in default_config.items():
    v_range=config[k].get('range')
    if v_range:
        #v_range = str(v_range).lower()
        default_config_v[k]=v_range.index(v)
    else:
        default_config_v[k]=v

  print('show default_config_v:')
  print(default_config_v)

  return default_config_v


async def main(test_config, os_setting, app_setting_hbase, app_setting_janusgraph):
    assert test_config.tune_os or test_config.tune_app, 'at least one of tune_app and tune_os should be True'
    app_setting_fix = get_default(app_setting_janusgraph)
    await optimize(test_config, os_setting, app_setting_hbase, app_setting_fix)
    print("hbase done")
    app_setting_fix = get_best_config(test_config)
    await optimize(test_config, os_setting, app_setting_janusgraph, app_setting_fix, sequence=False)
    print("janusgraph done")


async def optimize(test_config, os_setting, app_setting, app_setting_fix, sequence=True):
    # create optimizer
    optimizer = create_optimizer(
        test_config.optimizer.host,
        {
            **(os_setting if test_config.tune_os else {}),
            **(app_setting if test_config.tune_app else {}),
        },
        extra_vars=test_config.optimizer.extra_vars
    )
    if hasattr(optimizer, 'set_status_file'):
        optimizer.set_status_file(result_dir / 'optimizer_status')
    current_ob_x = []
    current_ob_y = []
    if sequence:
        task_id = -1
    else:
        task_id = test_config.optimizer.iter_limit
        test_config.optimizer.iter_limit = test_config.optimizer.iter_limit * 2 + 1
    tester, testee = test_config.hosts.tester, test_config.hosts.testee
    while task_id < test_config.optimizer.iter_limit:
        task_id += 1

        # - sample config
        if (task_id == 0) or (task_id == ((test_config.optimizer.iter_limit - 1)/2 + 1) and sequence == False):  # use default config
            sampled_config_numeric, sampled_config = None, get_default(app_setting)
        else:
            try:
                sampled_config_numeric, sampled_config = optimizer.get_conf(current_ob_x,current_ob_y)
            except StopIteration:
                # all configuration emitted
                return

        # - divide sampled config app & os
        sampled_os_config, sampled_app_config = divide_config(
            sampled_config,
            os_setting=os_setting,
            app_setting=app_setting
        )
        # if tune_app is off, just give sample_app_config a default value
        if test_config.tune_app is False:
            sampled_app_config = get_default(app_setting)

        # - dump configs
        os_config_path = result_dir / f'{task_id}_os_config.yml'
        os_config_path.write_text(
            yaml.dump(sampled_os_config, default_flow_style=False)
        )
        sampled_app_config = {**app_setting_fix, **sampled_app_config}
        app_config_path = result_dir / f'{task_id}_app_config.yml'
        app_config_path.write_text(
            yaml.dump(sampled_app_config, default_flow_style=False)
        )
        _print(f'{task_id}: os_config & app_config generated.')

        metric_results = []
        skip = False
        await single_deploy(
            task_name=test_config.task_name,
            task_id=task_id,
            tester=tester,
            testee=testee,
            tune_os=(task_id != 0 and test_config.tune_os),
            _skip=skip
        )
        
        for rep in range(test_config.optimizer.repitition):
            await single_test(
                task_name=test_config.task_name,
                task_id=task_id,
                rep=rep,
                tester=tester,
                testee=testee,
                tune_os=(task_id != 0 and test_config.tune_os),
                clients=test_config.clients,
                _skip=skip
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

            if result is None:
                for i in range(3):
                    await single_test(
                        task_name=test_config.task_name,
                        task_id=task_id,
                        rep=rep,
                        tester=tester,
                        testee=testee,
                        tune_os=(task_id != 0 and test_config.tune_os),
                        clients=test_config.clients,
                        _skip=skip
                    )
                    result = parse_result(
                        tester_name=test_config.tester,
                        result_dir=result_dir,
                        task_id=task_id,
                        rep=rep,
                        printer=_print
                    )
                    if result is not None:
                        break
                sys.exit(0)

            if result is not None and result != 0.:
                metric_results.append(result)
            _print(f'{task_id} - {rep}: done.')

        # after
        if (task_id != 0 and sequence == True) or (task_id != ((test_config.optimizer.iter_limit - 1)/2 + 1) and sequence == False):
            metric_result = - get_Kmeans_result(metric_results) if len(metric_results) > 0 else .0
            print("one step #########################################################\n")
            # add to bo
            # (X,y)
            # print(sampled_config_numeric, metric_result)
            optimizer.add_observation(
                (sampled_config_numeric, metric_result)
            )

            sampled_config_v = translate_config_to_numeric(sampled_config, app_setting)
            current_ob_x.append(sampled_config_v)
            current_ob_y.append(metric_result)
            # print(sampled_config_numeric)
            # print(metric_result)
            if hasattr(optimizer, 'dump_state'):
                optimizer.dump_state(result_dir / f'{task_id}_optimizer_state')

        if (task_id == 0) or (task_id == ((test_config.optimizer.iter_limit - 1)/2 + 1) and sequence == False):
            # y
            print("other step #########################################################\n")
            metric_result = - get_Kmeans_result(metric_results) if len(metric_results) > 0 else .0
            # default configs
            default_config = dict(get_default(app_setting))
            # default configs into numerics
            default_config_v = translate_config_to_numeric(default_config, app_setting)

            current_ob_x.append(default_config_v)
            current_ob_y.append(metric_result)

    # after reaching iter limit
    global proj_root


async def single_deploy(task_name, task_id, tester, testee, tune_os, _skip=False):
    global deploy_playbook_path
    global tester_playbook_path
    global osconfig_playbook_path

    # for debugging...
    if _skip:
        return

    # Hdconfigor/target/hbase/playbook/deploy.yml
    try:
        # - deploy db
        _print(f'{task_id}: deploying...')
        await run_playbook(
            deploy_playbook_path,
            host=testee,
            task_name=task_name,
            task_id=task_id,
        )

        if tune_os:
            # os parameters need to be changed
            _print(f'{task_id}: setting os parameters...')
            await run_playbook(
                osconfig_playbook_path,
                host=testee,
                task_name=task_name,
                task_id=task_id,
            )
        else:
            # - no need to change, for default testing or os test is configured to be OFF
            _print(f'{task_id}: resetting os parameters...')
            await run_playbook(
                osconfig_playbook_path,
                host=testee,
                task_name=task_name,
                task_id=task_id,
                tags='cleanup'
            )
        _print(f'{task_id}: done.')

        # - load data
        _print(f'{task_id}: load data...')
        await run_playbook(
            deploy_playbook_path2,
            host=tester,
            target=testee,
            task_name=task_name,
            task_id=task_id,
        )
        _print(f'{task_id}: done.')
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_error.log'
        errlog_path.write_text(str(e))
        print(e)


async def single_test(task_name, task_id, rep, tester, testee, tune_os, clients, _skip=False):
    global deploy_playbook_path
    global tester_playbook_path
    global osconfig_playbook_path

    # for debugging...
    if _skip:
        return

    _print(f'{task_id}: carrying out #{rep} repetition test...')
    # Hdconfigor/target/hbase/playbook/deploy.yml
    try:
        # - restart hbase
        _print(f'{task_id} - {rep}: restart hbase...')
        await run_playbook(
            deploy_playbook_path3,
            host=testee,
            task_name=task_name,
            task_id=task_id,
        )

        # - launch test and fetch result
        _print(f'{task_id} - {rep}: testing...')
        await run_playbook(
            tester_playbook_path,
            host=tester,
            target=testee,
            task_name=task_name,
            task_id=task_id,
            task_rep=rep,
        )
        _print(f'{task_id} - {rep}: done.')

    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_error_{rep}.log'
        errlog_path.write_text(str(e))
        print(e)



# -------------------------------------------------------------------------------------------------------
test_config = parse_cmd()
assert test_config is not None

# calculate paths
#  Hdconfigor
proj_root = Path(__file__, '../../..').resolve()
#  Hdconfigor/target/hbase
db_dir = proj_root / f'target/{test_config.target}'
result_dir = db_dir / f'results/{test_config.task_name}'
#  Hdconfigor/target/hbase/os_configs_info.yml
setting_path = proj_root / \
               f'target/{test_config.target}/os_configs_info.yml'
# Hdconfigor/target/hbase/playbook/deploy.yml
deploy_playbook_path = db_dir / 'playbook/deploy.yml'
deploy_playbook_path2 = db_dir / 'playbook/janusgraph_load.yml'
deploy_playbook_path3 = db_dir / 'playbook/restart_hbase.yml'
deploy_playbook_path4 = db_dir / 'playbook/close_hbase.yml'
# Hdconfigor/target/hbase/playbook/tester.yml
tester_playbook_path = db_dir / 'playbook/tester.yml'
# Hdconfigor/target/hbase/playbook/set_os.yml
osconfig_playbook_path = db_dir / 'playbook/set_os.yml'
# Hdconfigor/target/hbase/playbook/reboot.yml
reboot_playbook_path = db_dir / 'playbook/reboot.yml'
# Hdconfigor/target/hbase/workload/stress
workload_path = db_dir / f'workload/{test_config.workload}'
# Hdconfigor/target/hbase/os_configs_info.yml
os_setting_path = proj_root / \
                  f'target/{test_config.target}/os_configs_info.yml'
# Hdconfigor/target/hbase/app_configs_info.yml
setting_hbase = proj_root / \
                f'target/{test_config.target}/app_configs_info_hbase.yml'
setting_janusgraph = proj_root / \
                     f'target/{test_config.target}/app_configs_info_janusgraph.yml'

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
os_setting = yaml.load(os_setting_path.read_text())  # pylint: disable=E1101
app_setting_hbase = yaml.load(setting_hbase.read_text())  # pylint: disable=E1101
app_setting_janusgraph = yaml.load(setting_janusgraph.read_text())

# event loop, main() is async
loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(
        test_config=test_config,
        os_setting=os_setting,
        app_setting_hbase=app_setting_hbase,
        app_setting_janusgraph=app_setting_janusgraph
    )
)
loop.close()
