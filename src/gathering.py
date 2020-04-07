import requests
from utils import slice_url

class SimpleRequestGathering():

    def __init__(self, urls, **kwargs):
        self.urls =urls

    def collect(self):
        print("\nStart Collect\n")
        
        for labels, url in self.urls:
            print("Simple Request Collect: %s" % url)

            request = requests.get(url)
            _, domain, path = slice_url(url)
            
            yield {
                "fileId": "{}#{}".format(domain, path.replace("/", "-")),
                "url": url,
                "labels": labels,
                "text" : request.text 
            }

        print("\nFinishing Collect\n")
