"""
channel factory
"""


def create_bot(bot_type):
    """
    create a channel instance
    :param channel_type: channel type code
    :return: channel instance
    """
    bot_type_low = bot_type.lower()
    if bot_type_low == 'baidu':
        # Baidu Unit对话接口
        from bot.baidu.baidu_unit_bot import BaiduUnitBot
        return BaiduUnitBot()

    elif bot_type_low == 'chatgpt':
        # ChatGPT 网页端web接口
        from bot.chatgpt.chat_gpt_bot import ChatGPTBot
        return ChatGPTBot()

    elif bot_type_low == 'openai':
        # OpenAI 官方对话模型API
        from bot.openai.open_ai_bot import OpenAIBot
        return OpenAIBot()
    raise RuntimeError
