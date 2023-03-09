"""
channel factory
"""
import config
from channel.api.api_channel import ApiChannel
from channel.wechat.wechat_channel import WechatChannel


def create_channel(channel_type=None):
    """
    create a channel instance
    :param channel_type: channel type code
    :return: channel instance
    """
    if channel_type is None:
        channel_type = config.conf().get('channel_type')

    if channel_type == 'wx':
        return WechatChannel()
    if channel_type == 'api':
        return ApiChannel()
    raise RuntimeError
