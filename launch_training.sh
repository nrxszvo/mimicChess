name=default
nproc=1
nthread=8
cfg=cfg.yml
nnodes=1
node_rank=0
rdzv_id=456
rdzv_backend=c10d
rdzv_endpoint=0.0.0.0:1234
disable_commit=false

for var in "$@"
do
	param=(${var//=/ })
	if [[ ${param[0]} = "name" ]]
	then
		name=${param[1]}
	elif [[ ${param[0]} = "nproc" ]]
	then
		nproc=${param[1]}
	elif [[ ${param[0]} = "nthread" ]]
	then
		nthread=${param[1]}
	elif [[ ${param[0]} = "cfg" ]]
	then
		cfg=${param[1]}
	elif [[ ${param[0]} = "rdzv_endpoint" ]]
	then
		rdzv_endpoint=${param[1]}
	elif [[ ${param[0]} = "nnodes" ]]
	then
		nnodes=${param[1]}
	elif [[ ${param[0]} = "node_rank" ]]
	then
		node_rank=${param[1]}
	elif [[ ${param[0]} = "disable_commit" ]]
	then
		disable_commit=${param[1]}
	else
		echo "didn't recognize ${var}" 
		exit
	fi
done

if [ "$disable_commit" = false ]
then
	cmd="git diff-index --quiet HEAD --"
	$cmd
	if [ $? -ne 0 ]
	then
		echo "uncommited changes in working tree; exiting";
		exit
	fi
fi

commit=$(git rev-parse HEAD)
echo "commit: ${commit}"

cmd="OMP_NUM_THREADS=${nthread} torchrun --nnodes=${nnodes} --node-rank=${node_rank} --nproc-per-node=${nproc} train.py --name ${name} --commit ${commit} --cfg ${cfg} --num_nodes ${nnodes}"
echo $cmd
eval $cmd
