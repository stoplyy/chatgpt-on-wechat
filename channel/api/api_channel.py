# encoding:utf-8

"""
api channel
"""
import json
from flask import Flask, make_response, request
from channel.channel import Channel
from common.log import logger
import requests
import io

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/get/image')
def handler_single_msg(msg):
    msg = dict()
    msg['sessionUser'] = request.args.get('sessionUser', 'owner')
    msg['content'] = request.args.get('content', '')
    buffer = ApiChannel().handleImage(msg)
    binary_img = buffer.getvalue()
    response = make_response(binary_img)
    response.headers['Content-Type'] = 'image/png'
    return response


@app.route('/get/response')
def handler_response():
    msg = dict()
    msg['sessionUser'] = request.args.get('sessionUser', 'owner')
    msg['content'] = request.args.get('content', '')
    return ApiChannel().handle(msg)


class ApiChannel(Channel):
    def __init__(self):
        pass

    def startup(self):
        app.run()

    def handle(self, msg):
        logger.debug("receive msg: " + json.dumps(msg, ensure_ascii=False))
        from_user_id = msg['sessionUser']
        content = msg['content']
        res = self._do_send(content, from_user_id)
        return res

    def handleImage(self, msg):
        logger.debug("[WX]receive msg: " + json.dumps(msg, ensure_ascii=False))
        from_user_id = msg['sessionUser']
        content = msg['content']
        return self._do_send_img(content, from_user_id)

    def _do_send(self, query, reply_user_id):
        try:
            if not query:
                return
            context = dict()
            context['from_user_id'] = reply_user_id
            reply_text = super().build_reply_content(query, context)
            return reply_text
        except Exception as e:
            logger.exception(e)

    def _do_send_img(self, query, reply_user_id):
        try:
            if not query:
                return
            context = dict()
            context['type'] = 'IMAGE_CREATE'
            img_url = super().build_reply_content(query, context)
            if not img_url:
                return

            # 图片下载
            pic_res = requests.get(img_url, stream=True)
            image_storage = io.BytesIO()
            for block in pic_res.iter_content(1024):
                image_storage.write(block)
            image_storage.seek(0)

            # 图片发送
            logger.info('[API] sendImage, receiver={}'.format(reply_user_id))
            return image_storage
        except Exception as e:
            logger.exception(e)

    def check_prefix(self, content, prefix_list):
        for prefix in prefix_list:
            if content.startswith(prefix):
                return prefix
        return None

    def check_contain(self, content, keyword_list):
        if not keyword_list:
            return None
        for ky in keyword_list:
            if content.find(ky) != -1:
                return True
        return None
