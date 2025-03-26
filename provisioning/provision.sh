CONDA_VER=latest
OS_TYPE=$(uname -i)
CONDA_DIR=${HOME}/miniconda
PY_VER=3.10
cd ${HOME} 

if [ ! -d "${CONDA_DIR}" ]; then
	echo "removing local python site-packages..."
	rm -rf ${HOME}/.local/lib/python${PY_VER}/site-packages
	echo "installing conda..."
	curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
	bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p ${HOME}/miniconda -b
	rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
fi

if ! command -v conda 2>&1 >/dev/null
then
	export PATH=${HOME}/miniconda/bin:${PATH}
fi

conda update -y conda
conda init

alias clear=/usr/bin/clear

if [ ! -d "${HOME}/git" ]; then
	mkdir ${HOME}/git
fi

if [ ! -d "${HOME}/git/mimicChess" ]; then
	cd git
	git clone "https://${GHTOKEN}@github.com/nrxszvo/mimicChess.git"
	cd mimicChess
	git submodule set-url -- lib/pgnutils "https://${GHTOKEN}@github.com/nrxszvo/pgnutils.git"
	git submodule update --init --recursive

	if [ ! -e "datasets" ]; then
		datadir=$(ls ~ | grep mimicChess)
		ln -s ${HOME}/${datadir}/datasets .
		ln -s ${HOME}/${datadir}/models ckpts
	fi
	if [ -z ${MYNAME+x} ]; then
		echo "git name and email not specified; skipping git config"
	else
		git config --global user.name ${MYNAME} 
		git config --global user.email ${MYEMAIL} 
	fi
	cd ${HOME} 
fi

cd git/mimicChess
if [ "$OS_TYPE" = "aarch64" ]; then
	conda install -y anaconda::python=3.10
	sudo apt install gh
	gh auth login --with-token < ~/.ghtoken
	gh run download 12783586296 -R pytorch/pytorch --name manywheel-py3_10-cuda-aarch64
	pip install $(ls *.whl) 
	rm *.whl
	#gh run download 12787463616 -R pytorch/vision --name pytorch_vision__3.10_cu126_aarch64
	#pip install $(ls *.whl)
	#rm *.whl
	#gh run download 12787459535 -R pytorch/audio --name pytorch_audio__3.10_cu126_aarch64
	#pip install $(ls *.whl)
	#rm *.whl
	conda env update --file=environment_aarch64.yml
else
	conda env update --file=environment.yml
fi
cd ${HOME}

if ! command -v npm 2>&1 >/dev/null
then
	curl -fsSL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh
	sudo -E bash nodesource_setup.sh
	sudo apt-get install -y nodejs
fi
if ! command -v yarn 2>&1 > /dev/null
then
	sudo npm install --global yarn
fi

if [ ! -e "${HOME}/.vim/autoload/plug.vim" ]; then
	curl -fLo ${HOME}/.vim/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
	sudo add-apt-repository -y ppa:jonathonf/vim
	sudo apt update
	sudo apt install -y vim
fi

if [ ! -d "${HOME}/git/vimrc" ]; then
	cd ${HOME}/git
	git clone https://github.com/nrxszvo/vimrc.git
	cp vimrc/vimrc ${HOME}/.vimrc
	cp vimrc/coc-settings.json ${HOME}/.vim/coc-settings.json
fi

echo "set -g mouse on" > ${HOME}/.tmux.conf

sudo apt-get install -y stockfish libzstd-dev clangd

