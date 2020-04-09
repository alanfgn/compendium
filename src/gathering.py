import requests
from utils import slice_url

WRONG_ENCODING_SITES = ['www1.folha.uol.com.br']

class SimpleRequestGathering():

    def __init__(self, urls, **kwargs):
        self.urls =urls

    def collect(self):
        print("\nStart Simple Request")
        print("Collecting...\n")
        
        for labels, url in self.urls:
      
            request = requests.get(url)
            print("%s [%d]" % (url, request.status_code))

            _, domain, path = slice_url(url)
            
            if domain in WRONG_ENCODING_SITES:
                request.encoding = 'utf-8'

            yield {
                "fileId": "{}#{}".format(domain, path.replace("/", "-")),
                "url": url,
                "labels": labels,
                "text" : request.text 
            }

        print("\nFinishing Collect\n")
