# Object_Detection
 
## App for image segmentation on desktop devices
Simple app for using many segmentation image backends in easy way and test them with generic image segmenter.
<hr>

## How to install on Windows
You had to unclock running powershell scripts (for running python venv):
```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```
Create venv:
```PowerShell
python3.11 -m venv objectDetectionEnv
./objectDetectionEnv/Scripts/Activate.ps1

pip install -r requirements.txt
```
## How to install on Linux
Create venv:
```bash
python3.11 -m venv objectDetectionEnv
source ./objectDetectionEnv/bin/activate

pip install -r requirements.txt
```
### How to run:
```bash
python ./Object_Detection.py
```