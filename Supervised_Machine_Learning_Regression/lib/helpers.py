import requests


def download(url, filename):
    response = requests.get(url)
    
    print(f"content {response}")
    if response.status_code == 200:
        with open(filename,'wb') as f:
            f.write(response.content)