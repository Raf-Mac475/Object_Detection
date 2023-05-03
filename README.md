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

# How to import models

### I dont like LFS and github dont allows to keep files larger than 100MB without LFS - so lets keep them in a drive for a while. (maybe DVC in future)
Models to download:
```path
https://drive.google.com/drive/folders/1l6zO97RzdFzpmA0AwmTASTQ0psuBPP78?usp=share_link
```
After download you need to put them to `./models` directory. Sorry for nasty solutions. We will try to make it better in the future.
