if not exist ./data/ mkdir data
curl -L -o ./data/cinic10.zip https://www.kaggle.com/api/v1/datasets/download/mengcius/cinic10
tar -xf ./data/cinic10.zip -C ./data

if not exist ./.venv/ python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118