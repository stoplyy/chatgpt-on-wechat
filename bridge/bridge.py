from bot import bot_factory
import config

class Bridge(object):
    def __init__(self):
        pass

    def fetch_reply_content(self, query, context):
        bot_name = config.conf().get("bot_name")
        if bot_name is None:
            raise RuntimeError("bot_name is not set")
        return bot_factory.create_bot(bot_name).reply(query, context)
