import json
import base64
import urllib3

SERVER_ADDRESS = "http://localhost:8000/api/recognize"

class Client(object):

    def __init__(self, url):
        self.url = url

    def getBase64(self, filename):
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return encoded_string

    def request(self, api_func, request_data):
        url_func = "%s/api/%s" % (self.url, api_func)
        req = urllib2.Request(url=url_func, data = json.dumps(request_data), headers = {'content-type': 'application/json'})
        res = urllib2.urlopen(req)
        return res.read()

    def recognize(self, filename):
        base64Image = self.getBase64(filename)
        json_data = { "image" : base64Image }
        api_result = self.request("recognize", json_data)
        # print json.loads(api_result)

if __name__ == '__main__':

    client = Client(args.host)
    for image in args.images:
        client.recognize(image)

