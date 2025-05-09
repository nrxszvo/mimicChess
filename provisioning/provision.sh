if [ -d /mnt/disks/mimic ]; then
	HOME=/mnt/disks/mimic
fi
CONDA_VER=latest
OS_TYPE=$(uname -i)
if [[ "$OS_TYPE" == "unknown" ]]; then
	OS_TYPE=x86_64
fi
CONDA_DIR=${HOME}/miniconda
PY_VER=3.11

cd ${HOME} 

if [ ! -d "${CONDA_DIR}" ]; then
	if [ -d "${HOME}/.local/lib/python${PY_VER}/site-packages" ]; then
		rm -rf ${HOME}/.local/lib/python${PY_VER}/site-packages
	fi
	curl -LO "https://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
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
	git submodule set-url -- lib/dataset/cpp/zstd "https://github.com/facebook/zstd.git"
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

if command -v nvidia-smi 2>&1 >/dev/null 
then
	conda env create --name mimic --file=git/mimicChess/provisioning/environment_nvidia.yml
else
	conda env create --name mimic --file=git/mimicChess/provisioning/environment.yml
fi

sudo apt update
sudo apt install -y -V ca-certificates lsb-release wget stockfish libzstd-dev clangd tmux
wget https://packages.apache.org/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt update
sudo apt install -y -V libarrow-dev # For C++
sudo apt install -y -V libarrow-glib-dev # For GLib (C)
sudo apt install -y -V libarrow-dataset-dev # For Apache Arrow Dataset C++
sudo apt install -y -V libarrow-dataset-glib-dev # For Apache Arrow Dataset GLib (C)
sudo apt install -y -V libarrow-acero-dev # For Apache Arrow Acero
sudo apt install -y -V libarrow-flight-dev # For Apache Arrow Flight C++
sudo apt install -y -V libarrow-flight-glib-dev # For Apache Arrow Flight GLib (C)
sudo apt install -y -V libarrow-flight-sql-dev # For Apache Arrow Flight SQL C++
sudo apt install -y -V libarrow-flight-sql-glib-dev # For Apache Arrow Flight SQL GLib (C)
sudo apt install -y -V libgandiva-dev # For Gandiva C++
sudo apt install -y -V libgandiva-glib-dev # For Gandiva GLib (C)
sudo apt install -y -V libparquet-dev # For Apache Parquet C++
sudo apt install -y -V libparquet-glib-dev # For Apache Parquet GLib (C)
rm apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb

echo "set -g mouse on" > ${HOME}/.tmux.conf