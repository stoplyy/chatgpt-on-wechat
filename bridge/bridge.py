from bot import bot_factory
import config


class Bridge(object):
    def __init__(self):
        pass

    def fetch_reply_content(self, query, context):
        return bot_factory.create_bot(config.conf().get("bot_name")).reply(query, context)
