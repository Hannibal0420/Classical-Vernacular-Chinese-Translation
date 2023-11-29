import gdown

data_url = 'https://drive.google.com/drive/folders/1qSCrp_2zSjxFOk1PNxrLW29v_Z_R4mic?usp=share_link'
peft_config_url = 'https://drive.google.com/drive/folders/1YDP1ovyC3THxUxCABbxH0GCpF6Pc5owF?usp=share_link'

gdown.download_folder(data_url, quiet=True, use_cookies=False)
gdown.download_folder(peft_config_url, quiet=True, use_cookies=False)
