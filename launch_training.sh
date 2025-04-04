name=default
nproc=8
nthread=8
cfg=cfg.yml

cmd="git diff-index --quiet HEAD --"
$cmd
if [ $? -ne 0 ]
then
	echo "uncommited changes in working tree; exiting";
	exit
fi

commit=$(git rev-parse HEAD)
echo "commit: ${commit}"

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
	else
		echo "didn't recognize ${var}" 
		exit
	fi
done
cmd="OMP_NUM_THREADS=${nthread} torchrun --standalone --nnodes 1 --nproc-per-node ${nproc} train.py --name ${name} --commit ${commit} --cfg ${cfg}"
echo $cmd
eval $cmd
