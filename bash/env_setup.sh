#!/bin/bash
echo "[INFO] Current environment set to $1"

echo "Writing Kaggle API key to ~/.kaggle/kaggle.json"

mkdir -p ~/.kaggle
cat <<EOF > ~/.kaggle/kaggle.json
{"username":"atharvaingle","key":<your-key-here>}
EOF

cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle API Key successfully linked !!!"
pip3 install --upgrade --force-reinstall --no-deps kaggle
pip3 install --upgrade wandb
pip3 install seqeval
pip3 install --upgrade transformers datasets tokenizers
pip3 install -U rich
pip3 install python-dotenv
pip3 install slackclient
pip3 install sentencepiece
pip3 install hydra-core --upgrade
pip3 install -U scikit-learn
pip3 install pyarrow --upgrade

cd ~/.
if [ "$1" == "colab" ]
then
    echo "[INFO] Current environment -> $1"
    echo "Downloading data"
    cd /content/US-Patent-Matching-Kaggle
    mkdir input/
    cd input/
    kaggle datasets download -d atharvaingle/uspppm-data
    unzip uspppm-data.zip
    rm uspppm-data.zip
    echo "Data downloaded and unzipped successfully !!!"

elif [ "$1" == "kaggle" ]
then
    echo "[INFO] Current environment -> $1"
    echo "Downloading data"
    cd /kaggle/working/US-Patent-Matching-Kaggle
    mkdir input/
    cd input/
    kaggle datasets download -d atharvaingle/uspppm-data
    unzip uspppm-data.zip
    rm uspppm-data.zip
    echo "Data downloaded and unzipped successfully !!!"
else
    echo "⚠️ Unrecognized environment"
fi
