MNT_DIR=/home/web_server/mnt

host_file=doudizhu_host.txt
job_name=doudizhu
cmd=$1

((CUDA_VISIBLE_DEVICES=0))
((MAX_DEVICES=7))
DOCKER=docker


### config
PS_PORT=10000
WORKER_PORT=10001
RL_BUFFER_PORT=8333
SL_BUFFER_PORT=8334
SERVING_PORT=12001
NUM_PS=5
NUM_WOKER=17
TRAIN_GPU=7
# how many serving on one host
NUM_SERVING=7
# how many selfplay on one host
NUM_SELFPLAY=40

SAVED_MODEL_DIR=hdfs://default/home/rl/doudizhu/tmp/savedmodel
CHECKPOINT_DIR=hdfs://default/home/rl/doudizhu/tmp/checkpoint

if [ ! -e $host_file ] ; then
  echo "host file $host_file does not exist"
  exit 1
fi

# rl buffer urls and sl buffer urls
readarray -t hosts < $host_file
rl_buffer_urls=tcp://${hosts[0]}:$RL_BUFFER_PORT
sl_buffer_urls=tcp://${hosts[0]}:$SL_BUFFER_PORT
for ((i=1; i < ${#hosts[@]}; i++))
do
  rl_buffer_urls=$rl_buffer_urls,tcp://${hosts[$i]}:$RL_BUFFER_PORT
  sl_buffer_urls=$sl_buffer_urls,tcp://${hosts[$i]}:$SL_BUFFER_PORT
done

echo rl_buffer_urls: $rl_buffer_urls
echo sl_buffer_urls: $sl_buffer_urls

# cluster info
ps=""
worker=""
chief=${hosts[0]}:$WORKER_PORT
for ((i=0; i < ${#hosts[@]}; i++))
do
  if [ $i -lt $NUM_PS ]
  then
    if [ $i -eq 0 ] ; then
      ps=${hosts[$i]}:$PS_PORT
    else
      ps=$ps,${hosts[$i]}:$PS_PORT
    fi
  fi
  if [ $i -lt $NUM_WOKER ]
  then
    if [ $i -eq 1 ] ; then
      worker=${hosts[$i]}:$WORKER_PORT
    else
      worker=$worker,${hosts[$i]}:$WORKER_PORT
    fi
  fi
done
echo ps: $ps
echo worker: $worker
echo chief: $chief

# serving urls
serving_url="127.0.0.1:$SERVING_PORT"
for ((i=1; i < $NUM_SERVING; i++)) ; do
  serving_url=$serving_url,127.0.0.1:$(($SERVING_PORT + $i))
done
echo serving_url: $serving_url

all_host=$(cat $host_file)
docker_tensor_args="--rm --net=host --runtime nvidia"
node_image=registry.corp.kuaishou.com/aiplatform/node:6-stretch
serving_image=10.62.216.44/library/tensorflowserving:hdfs-base4
train_docker_image=registry.corp.kuaishou.com/aiplatform/tensorflow:1.8.0-gpu-py3-hdfsrl

function sync {
  for host in $all_host ; do
    rsync -a -l --delete $MNT_DIR/hundun $host:$MNT_DIR &
    rsync -a -l --delete $MNT_DIR/nfspgame $host:$MNT_DIR &
  done
  wait
}

function start_train {
  train_docker_args=" --rm -d \
    --net host \
    -v /home/web_server/mnt/:/mnt \
    -w /mnt/hundun/hundun/src/scripts \
    -e PYTHONPATH=/mnt/hundun/hundun/src \
    -e RL_BUFFER_URL=tcp://0.0.0.0:$RL_BUFFER_PORT -e SL_BUFFER_URL=tcp://0.0.0.0:$SL_BUFFER_PORT \
    -e PS=$ps -e WORKER=$worker -e CHIEF=$chief \
    -e CHECKPOINT_DIR=$CHECKPOINT_DIR \
    -e EXPORT_DIR_BASE=$SAVED_MODEL_DIR \
    $train_docker_image "
  for ((i=0; i < ${#hosts[@]}; i++)) ; do
    host=${hosts[$i]}
    if [ $i -lt $NUM_PS ] 
    then
      # start ps
      ssh $host $DOCKER container run -e CUDA_VISIBLE_DEVICES="" \
          --name ${job_name}_train_ps \
          $train_docker_args python3 run.py --task_type ps --task_index $i --config nfsp.ini && echo start ps &
    fi
    if [ $i -lt $NUM_WOKER ]
    then
      # start chief
      if [ $i -eq 0 ]
      then
      ssh $host $DOCKER container run -e CUDA_VISIBLE_DEVICES=$TRAIN_GPU \
          --name ${job_name}_train \
          $train_docker_args python3 run.py --task_type chief --task_index 0 --config nfsp.ini &
      else
        # start worker
        ssh $host $DOCKER container run -e CUDA_VISIBLE_DEVICES=$TRAIN_GPU \
            --name ${job_name}_train \
            $train_docker_args python3 run.py --task_type worker --task_index $(($i - 1)) --config nfsp.ini &
      fi
    fi
  done
  wait
}

function run_command {
  train_docker_args=" --rm \
    -v /home/web_server/:/device \
    $train_docker_image "
  host_id=0
  for host in $all_host ; do
      ssh $host $DOCKER container run \
          $train_docker_args rm -rf /device/mnt/
  done
}

function kill_train {
  for ((i=0; i < ${#hosts[@]}; i++)) ; do
    host=${hosts[$i]}
    if [ $i -lt $NUM_PS ] 
    then
      ssh $host "$DOCKER kill ${job_name}_train_ps " && echo "done kill docker doudizhu_train_ps @ $host" &
    fi
    if [ $i -lt $NUM_WOKER ]
    then
      ssh $host "$DOCKER kill ${job_name}_train " && echo "done kill docker doudizhu_train @ $host" &
    fi
  done
  wait
}

function start_selfplay {
  set -x
  for host in $all_host ; do
    echo $host
    docker_selfplay_args="--network host -v /home/web_server/mnt/:/mnt --rm -w /mnt/nfspgame/src/scripts -d \
      -e CLIENTS_URL=$serving_url -e RL_BUFFER_URL=$rl_buffer_urls -e SL_BUFFER_URL=$sl_buffer_urls"
    for((id=0; id<$NUM_SELFPLAY; id++)) ; do
      curr_docker_args="$docker_selfplay_args --name ${job_name}_selfplay_$id"
      cmd="/usr/local/bin/node run.js ./examples/Landlords.json"
      ssh $host "$DOCKER container run $curr_docker_args $node_image $cmd && echo start docker @ $host" &
    done
   done
   wait
}

function kill_selfplay {
  for host in $all_host ; do
    for((id=0; id<$NUM_SELFPLAY; id++)) ; do
      ssh $host "$DOCKER kill ${job_name}_selfplay_$id" && echo " kill ${job_name}_selfplay_$id @ $host" &
    done
   done
   wait
}

function start_serving {
  for host in $all_host ; do
    ((port=$SERVING_PORT))
    docker_serving_args="--network host -d --rm "
    for((id=0; id<$NUM_SERVING; id++)) ; do
      curr_docker_args="$docker_serving_args -e CUDA_VISIBLE_DEVICES=$id --name ${job_name}_serving_$id"
      cmd="tensorflow_model_server --port=$port --enable_batching=true --model_base_path=$SAVED_MODEL_DIR"
      ((CUDA_VISIBLE_DEVICES=(CUDA_VISIBLE_DEVICES+1)%MAX_DEVICES))
      ((port=port+1))
      ssh $host "$DOCKER container run $curr_docker_args $serving_image $cmd && echo start docker @ $host" &
    done
   done
   wait
}

function kill_serving {
  for host in $all_host ; do
    for((id=0; id<$NUM_SERVING; id++)) ; do
      ssh $host "$DOCKER kill ${job_name}_serving_$id" && echo " kill ${job_name}_serving_$id @ $host" &
    done
   done
   wait
}


function start_local_serving {
    docker_args="--network host -d --rm -e CUDA_VISIBLE_DEVICES=0 --name ${job_name}_local_serving"
    cmd="tensorflow_model_server --port=$SERVING_PORT --enable_batching=true --model_base_path=$SAVED_MODEL_DIR"
    $DOCKER container run $docker_args $serving_image $cmd && echo "start local docker"
}

function kill_local_serving {
    $DOCKER kill ${job_name}_local_serving && echo "kill ${job_name}_local_serving"
}

function kill_all_docker {
  for host in $all_host ; do
    ssh $host 'docker kill $(docker ps -q)' &
   done
   wait
}

function remove_checkpoint {
  docker container run  --rm --net=host 0b998739e971 /home/hadoop/bin/hadoop fs -rm -r $(echo $SAVED_MODEL_DIR  | sed -e 's/hdfs:\/\/default//')
  docker container run  --rm --net=host 0b998739e971 /home/hadoop/bin/hadoop fs -rm -r $(echo $CHECKPOINT_DIR |sed -e 's/hdfs:\/\/default//')
}

# sync whenever run the script
sync

case $cmd in
  sync)
    echo "done"
    ;;
  start_serving)
    start_serving
    ;;
  kill_serving)
    kill_serving
    ;;
  start_train)
    start_train
    ;;
  kill_train)
    kill_train;
    ;;
  start_selfplay)
    start_selfplay
    ;;
  kill_selfplay)
    kill_selfplay
    ;;
  cmd)
    run_command "$args"
    ;;
  start_all)
    start_train 
    sleep 10
    start_serving
    sleep 20
    start_selfplay 
    ;;
  restart_selfplay)
    kill_selfplay
    start_selfplay
    ;;
  kill_all)
    kill_train &
    kill_serving &
    kill_selfplay &
    wait
    ;;
  kill_all_docker)
    kill_all_docker
    ;;
  remove_checkpoint)
    remove_checkpoint
    ;;
  start_local_serving)
    start_local_serving
    ;;
  kill_local_serving)
    kill_local_serving
    ;;
  *)
    echo "unkown cmd $cmd"
    exit 1
esac
