if [ ! -d "${HOME}/git" ]; then
	mkdir ${HOME}/git
fi

if [ ! -d "${HOME}/git/mimicChess" ]; then
	cd git
	git clone "https://${GHTOKEN}@github.com/nrxszvo/mimicChess.git"
	git clone "https://${GHTOKEN}@github.com/nrxszvo/pzp.git"

	if [ -z ${MYNAME+x} ]; then
		echo "git name and email not specified; skipping git config"
	else
		git config --global user.name ${MYNAME} 
		git config --global user.email ${MYEMAIL} 
	fi
	cd ${HOME} 
fi

cd git/mimicChess

python -m venv env
. env/bin/activate
if command -v nvidia-smi 2>&1 >/dev/null 
then
	pip install -r provisioning/requirements_nvidia.txt
fi
pip install -r provisioning/requirements.txt

cd ${HOME}

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
