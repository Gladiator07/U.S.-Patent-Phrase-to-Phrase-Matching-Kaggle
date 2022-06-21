#!/bin/bash
cd ../

printf "Checking code changes in main repo's source code\n"
git pull

printf "\nGoing to .sync_code directory\n"
cd .sync_code/US-Patent-Matching-Kaggle

printf "\n Pulling changes to kaggle dataset's code\n"
echo "Pulling latest source code"
git pull

cd ..
kaggle datasets version -p ./ -m "update" --dir-mode zip