import asyncio
import os
import sys
from statistics import mean

import yaml
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.cluster import KMeans
from lib.other import parse_cmd, run_playbook, get_default
from lib.optimizer import create_optimizer
import math


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


def ansible_hosts_config(nodes):
    with open('/etc/ansible/hosts', 'w') as f:
        f.write("[redis-test]\n")
        for node in nodes:
            ansible_config = "{} ansible_ssh_host={} ansible_ssh_user='{}' ansible_ssh_pass='{}' " \
                             "ansible_become_pass='{}' ansible_ssh_port=22 " \
                             "intranet_ip={}". \
                format(node["name"], node["ip"], node["user"], node["password"], node["password"], node["ip"])
            f.write(ansible_config + "\n")


def getResult(csv_path):
    result = {}
    file_exist = os.path.exists(csv_path)
    if not file_exist:
        return None
    with open(csv_path, "r") as f:
        content = f.read().split("\n")
        key = content[0].split(",")
        value = content[1].split(",")
        for i in range(len(key)):
            result[key[i]] = value[i]
    return result


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


def divide_config(sampled_config):
    for k in sampled_config.keys():
        if type(sampled_config[k]) is bool:
            # make sure no uppercase 'True/False' literal in result
            sampled_config[k] = str(sampled_config[k]).lower()
        elif type(sampled_config[k]) is np.float64:
            sampled_config[k] = float(sampled_config[k])
    return sampled_config


def _print(msg):
    print(f'[{datetime.now()}] {test_config.task_name} - {msg}')
    # print('[' + datetime.now() + ']')


def score_function(recordsPerInterval, real_throughput, latency_threshold, real_latency, hyper_parameter, cost):
    score = hyper_parameter * min(1.0, latency_threshold / real_latency) * (real_throughput / (recordsPerInterval * 1000)) \
            + (1 - hyper_parameter) / cost
    return score * 100000


def getScore(test_config, i, j, cost):
    result_path = "../../target/{}/results/{}/{}_run_result_{}.csv". \
        format(test_config.target, test_config.task_name, i, j)
    performance_metrics = getResult(result_path)
    if performance_metrics:
        throughput = float(performance_metrics[test_config.throughput])
        mean_latency = float(performance_metrics[test_config.mean_latency])
        # p99_latency = float(performance_metrics[test_config.p99_latency])
        # 5表示2 core 4G cost:5块/h  4core 8G cost:10块/
        if throughput == 0:
            score = 0
        else:
            score = score_function(test_config.optimizer.recordsPerInterval, throughput,
                                   test_config.latency_threshold, mean_latency,
                                   test_config.score_hyper_parameter, cost)
        return score
    return 0


async def main(test_config, init_id, app_setting):
    global proj_root

    # create optimizer
    optimizer = create_optimizer(
        test_config.optimizer.name,
        {
            **app_setting,
        },
        extra_vars=test_config.optimizer.extra_vars
    )

    if hasattr(optimizer, 'set_status_file'):
        optimizer.set_status_file(result_dir / 'optimizer_status')
    task_id = init_id
    while task_id < test_config.optimizer.iter_limit:
        task_id += 1

        # - sample config
        if task_id == 0:  # use default config
            sampled_config_numeric, sampled_config = None, get_default(app_setting)
            sampled_config_numeric = translate_config_to_numeric(sampled_config, app_setting)
        else:
            try:
                sampled_config_numeric, sampled_config = optimizer.get_conf()
            except StopIteration:
                # all configuration emitted
                return

        print(sampled_config)

        # 格式转换
        sampled_app_config = divide_config(sampled_config)
        # if tune_app is off, just give sample_app_config a default value

        # 生成配置
        app_config_path = result_dir / f'{task_id}_app_config.yml'
        app_config_path.write_text(
            yaml.dump(sampled_app_config, default_flow_style=False)
        )
        app_config_pitch_path = result_dir / f'{task_id}_app_config_pitch.yml'
        app_config_pitch_path.write_text(
            yaml.dump(sampled_config_numeric, default_flow_style=False)
        )
        _print(f'{task_id}: app_config generated.')

        metric_results = []

        qua_sum = 0
        for k, v in sampled_config_numeric.items():
            qua_sum += int(v) ** 2
        result = - (math.cos(qua_sum ** 1 / 2 * 12) + 1) / (0.5 * qua_sum + 2)

        print(result)
        metric_results.append(result)

        metric_result = mean(metric_results) if len(metric_results) > 0 else 0

        if task_id != 0:
            optimizer.add_observation((sampled_config_numeric, metric_result))
            if hasattr(optimizer, 'dump_state'):
                optimizer.dump_state(result_dir / f'{task_id}_optimizer_state')



    # after reaching iter limit


# 将VM的信息输出出来
async def generate_vms(master, task_name, task_id):
    global generate_VMs_playbook_path

    try:
        # generate_VMs
        _print(f'{task_id}: start generate_VMs...')
        stdout, stderr = await run_playbook(
            generate_VMs_playbook_path,
            master=master,
            task_name=task_name,
            task_id=task_id
        )
        out_log_path = result_dir / f'{task_id}_generate_VMs_out.log'
        out_log_path.write_text(stdout)
        _print(f'{task_id}: generate_VMs done.')
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_generate_VMs_error.log'
        errlog_path.write_text(str(e))
        print(e)


async def prepare_vms(VMs, task_name, task_id):
    global prepare_VMs_playbook_path

    try:
        # prepare_vms
        _print(f'{task_id}: start prepare vms...')
        stdout, stderr = await run_playbook(
            prepare_VMs_playbook_path,
            VMs=VMs,
            task_name=task_name,
            task_id=task_id
        )
        out_log_path = result_dir / f'{task_id}_prepare_VMs_out.log'
        out_log_path.write_text(stdout)
        _print(f'{task_id}: prepare_VMs done.')
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_prepare_VMs_error.log'
        errlog_path.write_text(str(e))
        print(e)


async def deploy_clusters(task_name, task_id, flink_vms, kafka_vms):
    global deploy_playbook_flink_path
    global deploy_playbook_zookeeper_path
    global deploy_playbook_kafka_path

    try:
        # - deploy flink
        _print(f'{task_id}: deploying clusters...')
        _print(f'{task_id}: deploying hadooop and flink ...')
        stdout, stderr = await run_playbook(
            deploy_playbook_flink_path,
            VMs=flink_vms,
            task_name=task_name,
            task_id=task_id
        )
        out_log_path = result_dir / f'{task_id}_deploy_flink_out.log'
        out_log_path.write_text(stdout)
        _print(f'{task_id}: deploying hadooop and flink done.')

        # - deploy zookeeper
        _print(f'{task_id}: deploying zookeeper...')
        stdout, stderr = await run_playbook(
            deploy_playbook_zookeeper_path,
            VMs=kafka_vms,
            task_name=task_name,
            task_id=task_id
        )
        out_log_path = result_dir / f'{task_id}_deploy_zookeeper_out.log'
        out_log_path.write_text(stdout)
        _print(f'{task_id}: deploying zookeeper done.')

        # - deploy kafka
        # 循环部署
        _print(f'{task_id}: deploying kafka...')
        stdout, stderr = await run_playbook(
            deploy_playbook_kafka_path,
            VMs=kafka_vms,
            task_name=task_name,
            task_id=task_id
        )
        out_log_path = result_dir / f'{task_id}_deploy_kafka_out.log'
        out_log_path.write_text(stdout)
        _print(f'{task_id}: deploying kafka done.')
        _print(f'{task_id}: deploying clusters done.')
    except RuntimeError as e:
        outlog_path = result_dir / f'{task_id}_deploying_clusters_error.log'
        outlog_path.write_text(str(e))
        print(e)


async def vm_aftercure(VMs, master, task_id):
    global VMs_aftercure_playbook_path

    try:
        stdout, stderr = await run_playbook(
            VMs_aftercure_playbook_path,
            VMs=VMs,
            master=master
        )
        out_log_path = result_dir / f'{task_id}_vm_aftercure_out.log'
        out_log_path.write_text(stdout)
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_vm_aftercure_error.log'
        errlog_path.write_text(str(e))
        print(e)


async def clean_vm(VMs, task_id):
    global clean_VMs_playbook_path

    try:
        for vm in VMs:
            stdout, stderr = await run_playbook(
                clean_VMs_playbook_path,
                master=vm
            )
        out_log_path = result_dir / f'{task_id}_clean_vm_out.log'
        out_log_path.write_text(stdout)
    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_clean_vm_error.log'
        errlog_path.write_text(str(e))
        print(e)


async def single_test(master, kafka_vms, workload, task_name, task_id, rep, recordsPerInterval, _skip=False):
    global tester_playbook_path

    # for debugging...
    if _skip:
        return

    try:
        _print(f'{task_id} - {rep}: testing...')
        stdout, stderr = await run_playbook(
            tester_playbook_path,
            VMs=kafka_vms,
            master=master,
            workload=workload,
            task_name=task_name,
            task_id=task_id,
            task_rep=rep,
            recordsPerInterval=recordsPerInterval
        )
        out_log_path = result_dir / f'{task_id}_test_out.log'
        out_log_path.write_text(stdout)
        _print(f'{task_id} - {rep}: test done.')

    except RuntimeError as e:
        errlog_path = result_dir / f'{task_id}_test_error_{rep}.log'
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

generate_VMs_playbook_path = db_dir / 'playbook/generate_VMs.yml'
prepare_VMs_playbook_path = db_dir / 'playbook/prepare_VMs.yml'
deploy_playbook_flink_path = db_dir / 'playbook/deploy_flink.yml'
deploy_playbook_zookeeper_path = db_dir / 'playbook/deploy_zookeeper.yml'
deploy_playbook_kafka_path = db_dir / 'playbook/deploy_kafka.yml'
VMs_aftercure_playbook_path = db_dir / 'playbook/vms_aftercure.yml'
clean_VMs_playbook_path = db_dir / 'playbook/clean_VMs.yml'
tester_playbook_path = db_dir / 'playbook/tester.yml'

app_setting_path = proj_root / f'target/{test_config.target}/app_configs_info.yml'

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
app_setting = yaml.load(app_setting_path.read_text())  # pylint: disable=E1101

# event loop, main() is async
loop = asyncio.get_event_loop()
loop.run_until_complete(
    main(
        test_config=test_config,
        init_id=init_id,
        app_setting=app_setting
    )
)
loop.close()
