import requests
from utils import slice_url

class SimpleRequestGathering():

    def __init__(self, urls, **kwargs):
        self.urls =urls

    def collect(self):
        for labels, url in self.urls:
            
            request = requests.get(url)
            _, domain, path = slice_url(url)
            
            yield {
                "fileId": "{}#{}".format(domain, path.replace("/", "-")),
                "url": url,
                "labels": labels,
                "text" : request.text 
            }
