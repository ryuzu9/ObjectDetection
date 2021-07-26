from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

import http.client, urllib.request, urllib.parse, urllib.error, base64, json, io, tempfile, requests, cv2

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def imread_web(url):
    res = requests.get(url)
    img = None
    with tempfile.NamedTemporaryFile(dir='./') as fp:
        fp.write(res.content)
        fp.file.seek(0)
        img = cv2.imread(fp.name)
    return img

if __name__ == "__main__":
    # 以下の情報については、各自の環境に応じて変更する。
    subscription_key = ""
    endpoint = ""

    headers = {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    params = urllib.parse.urlencode({
        'visualFeatures': 'Objects',
        'language': 'en',
    })

    # 好きな画像のURLに書き換える
    # 物体の全体が映っている写真(JPEG, PNGなど)
    image_url = 'http://blog-imgs-36.fc2.com/d/r/i/drinkactman/PICT0126A.jpg'

    body = { 'url': image_url }
    body = json.dumps(body)

    try:
        conn = http.client.HTTPSConnection('takano0624.cognitiveservices.azure.com')
        conn.request("POST", "/vision/v2.1/analyze?%s" % params, body, headers)
        response = conn.getresponse()
        data = json.loads(response.read())
        print(json.dumps(data, indent=2))
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

    raw_img = imread_web(image_url)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    json_dict = json.loads(json.dumps(data, indent=2))
    for item in json_dict["objects"]:
        x = item["rectangle"]["x"]
        y = item["rectangle"]["y"]
        w = item["rectangle"]["w"]
        h = item["rectangle"]["h"]
        edited_img = cv2.putText(raw_img, item["object"], (item["rectangle"]["x"]+5, item["rectangle"]["y"]+35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3, cv2.LINE_AA)
        edited_img = cv2.rectangle(edited_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    plt.figure(figsize=(3, 3), dpi=300)
    plt.imshow(edited_img)
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)    
    plt.show()
