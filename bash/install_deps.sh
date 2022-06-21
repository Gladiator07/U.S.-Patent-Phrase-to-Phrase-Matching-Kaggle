# setup google cloud
apt-get install apt-transport-https ca-certificates gnupg -y
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
apt-get update && apt-get install google-cloud-cli -y

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
