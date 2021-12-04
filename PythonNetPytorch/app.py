from flask import Flask, request
import base64
from io import BytesIO
import fasterrcnnService as Service
import pyheif
from PIL import Image

app = Flask(__name__)


@app.route('/detect/imageDetect', methods=['post'])
def upload():
    # step 1. receive image
    file = request.form.get('imageBase64Code')

    try:
        # 对于普通图片的检测
        image = Image.open(BytesIO(base64.b64decode(file)))
    except(Exception) :
        # ios设备上传图片格式是HEIC，需要特殊处理，先转成png格式
        with open('./images_buffer/convert.HEIC', 'wb') as f:
            f.write(base64.b64decode(file))
        file_path = './images_buffer/convert.HEIC'
        data = read_image_file_rb(file_path)
        i = pyheif.read_heif(data)
        pi = Image.frombytes(mode=i.mode, size=i.size, data=i.data)
        pi.save('./images_buffer/convert.jpg', format="jpeg")
        image = Image.open('./images_buffer/convert.jpg')

    # step 2. detect image
    dec_time = Service.detect(image)
    print("inference+NMS time: {}".format(dec_time))

    # step 3. convert image to byte_array
    with open("./images_buffer/new_test.jpg", 'rb') as f:
        base64_data = base64.b64encode(f.read())
        detect_out = base64_data.decode()
    return detect_out


def read_image_file_rb(file_path):
    with open(file_path, 'rb') as f:
        file_data = f.read()
    return file_data


if __name__ == '__main__':
    app.run(port=5000, debug=True)
