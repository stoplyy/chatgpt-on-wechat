"""
A simple wrapper for the official ChatGPT API
"""
import argparse
import json
import os
import sys
from datetime import date

import openai
import tiktoken

from bot.bot import Bot
from config import conf

ENGINE = os.environ.get("GPT_ENGINE") or conf().get("gpt_model_name") or "text-chat-davinci-002-20221122"

ENCODER = tiktoken.encoding_for_model(ENGINE)

APIBASE =os.environ.get("api_url") or conf().get("api_url") or "https://api.openai.customdomain.com/v1/chat/completions"

def get_max_tokens(prompt: str) -> int:
    """
    Get the max tokens for a prompt
    """
    return 4000 - len(ENCODER.encode(prompt))


# ['text-chat-davinci-002-20221122']
class Chatbot:
    """
    Official ChatGPT API
    """

    def __init__(self, api_key: str, buffer: int = None) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # openai.api_base = APIBASE
        self.base_prompt ={"role": "system", "content": os.environ.get("CUSTOM_BASE_PROMPT") or "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally. "}
        self.conversations = Conversation()
        self.prompt = Prompt(buffer=buffer)

    def _get_completion(
            self,
            prompt: list,
            temperature: float = 0.5,
            stream: bool = False,
    ):
        """
        Get the completion function
        """
        prompt.insert(0, self.base_prompt)
        return openai.ChatCompletion.create(
            model=ENGINE,
            messages=prompt
        )

    def _process_completion(
            self,
            user_request: str,
            completion: dict,
            conversation_id: str = None,
            user: str = "User",
    ) -> dict:
        # completion.choices[0].message.content
        if completion.get("choices") is None:
            raise Exception("ChatGPT API returned no choices")
        if len(completion["choices"]) == 0:
            raise Exception("ChatGPT API returned no choices")
        if completion["choices"][0].get("message") is None:
            raise Exception("ChatGPT API returned no text")
        if completion["choices"][0]["message"].get("content") is None:
            raise Exception("ChatGPT API returned no text")
        
        content = completion["choices"][0]["message"]["content"]
        # Add to chat history
        self.prompt.add_to_history(
            user_request,
            content,
            user=user,
        )
        if conversation_id is not None:
            self.save_conversation(conversation_id)
            
        return content

    def _process_completion_stream(
            self,
            user_request: str,
            completion: dict,
            conversation_id: str = None,
            user: str = "User",
    ) -> str:
        full_response = ""
        for response in completion:
            if response.get("choices") is None:
                raise Exception("ChatGPT API returned no choices")
            if len(response["choices"]) == 0:
                raise Exception("ChatGPT API returned no choices")
            if response["choices"][0].get("finish_details") is not None:
                break
            if response["choices"][0].get("text") is None:
                raise Exception("ChatGPT API returned no text")
            if response["choices"][0]["text"] == "<|im_end|>":
                break
            yield response["choices"][0]["text"]
            full_response += response["choices"][0]["text"]

        # Add to chat history
        self.prompt.add_to_history(user_request, full_response, user)
        if conversation_id is not None:
            self.save_conversation(conversation_id)

    def ask(
            self,
            user_request: str,
            temperature: float = 0.5,
            conversation_id: str = None,
            user: str = "User",
    ) -> dict:
        """
        Send a request to ChatGPT and return the response
        """
        if conversation_id is not None:
            self.load_conversation(conversation_id)
            
        completion = self._get_completion(
            self.prompt.construct_prompt(user_request, user=user),
            temperature,
        )
        return self._process_completion(user_request, completion, user=user)

    def ask_stream(
            self,
            user_request: str,
            temperature: float = 0.5,
            conversation_id: str = None,
            user: str = "User",
    ) -> str:
        """
        Send a request to ChatGPT and yield the response
        """
        if conversation_id is not None:
            self.load_conversation(conversation_id)
        prompt = self.prompt.construct_prompt(user_request, user=user)
        return self._process_completion_stream(
            user_request=user_request,
            completion=self._get_completion(prompt, temperature, stream=True),
            user=user,
        )

    def make_conversation(self, conversation_id: str) -> None:
        """
        Make a conversation
        """
        self.conversations.add_conversation(conversation_id, [])

    def rollback(self, num: int) -> None:
        """
        Rollback chat history num times
        """
        for _ in range(num):
            self.prompt.chat_history.pop()

    def reset(self) -> None:
        """
        Reset chat history
        """
        self.prompt.chat_history = []

    def load_conversation(self, conversation_id) -> None:
        """
        Load a conversation from the conversation history
        """
        if conversation_id not in self.conversations.conversations:
            # Create a new conversation
            self.make_conversation(conversation_id)
        self.prompt.chat_history = self.conversations.get_conversation(conversation_id)

    def save_conversation(self, conversation_id) -> None:
        """
        Save a conversation to the conversation history
        """
        self.conversations.add_conversation(conversation_id, self.prompt.chat_history)


class AsyncChatbot(Chatbot):
    """
    Official ChatGPT API (async)
    """

    async def _get_completion(
            self,
            prompt: list,
            temperature: float = 0.5,
            stream: bool = False,
    ):
        """
        Get the completion function
        """
        return openai.ChatCompletion.acreate(
            model=ENGINE,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant",
                    "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        )
        # return openai.ChatCompletion.acreate(
        #     engine=ENGINE,
        #     prompt=prompt,
        #     temperature=temperature,
        #     max_tokens=get_max_tokens(prompt),
        #     stop=["\n\n\n"],
        #     stream=stream,
        # )

    async def ask(
            self,
            user_request: str,
            temperature: float = 0.5,
            user: str = "User",
    ) -> dict:
        """
        Same as Chatbot.ask but async
        }
        """
        completion = await self._get_completion(
            self.prompt.construct_prompt(user_request, user=user),
            temperature,
        )
        return self._process_completion(user_request, completion, user=user)

    async def ask_stream(
            self,
            user_request: str,
            temperature: float = 0.5,
            user: str = "User",
    ) -> str:
        """
        Same as Chatbot.ask_stream but async
        """
        prompt = self.prompt.construct_prompt(user_request, user=user)
        return self._process_completion_stream(
            user_request=user_request,
            completion=await self._get_completion(prompt, temperature, stream=True),
            user=user,
        )


class Prompt:
    """
    Prompt class with methods to construct prompt
    """

    def __init__(self, buffer: int = None) -> None:
        """
        Initialize prompt with base prompt
        """
        # Track chat history
        self.chat_history: list = []
        self.buffer = buffer

    def add_to_history(
            self,
            user_request: str,
            response: str,
            user: str = "User",
    ) -> None:
        """
        Add request/response to chat history for next prompt
        """
        self.chat_history.extend([{"role": "user", "content": user_request}, {"role": "assistant", "content": response}])

    def history(self, custom_history: list = None) -> str:
        """
        Return chat history
        """
        return "\n".join(custom_history or self.chat_history)
    
    def get_token_count(self, list: str = "default") -> int:
            """
            Get token count
            """
            if ENGINE not in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]:
                raise NotImplementedError("Unsupported engine {ENGINE}")

            num_tokens = 0
            for message in list:
                # every message follows <im_start>{role/name}\n{content}<im_end>\n
                num_tokens += 4
                for key, value in message.items():
                    num_tokens += len(ENCODER.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens 
        
    def construct_prompt(
            self,
            new_prompt: str,
            user: str = "User",
    ) -> str:
        """
        Construct prompt based on chat history and request
        """
        prompt_list = [{"role": "user", "content": new_prompt}]
        prompt_list.extend(self.chat_history)
        
        # Check if prompt over 4000*4 characters
        if self.buffer is not None:
            max_tokens = 4000 - self.buffer
        else:
            max_tokens = 3200
        if self.get_token_count(prompt_list) > max_tokens:
            # Remove oldest chat
            if len(self.chat_history) == 0:
                return prompt_list
            # remove first req&resp
            self.chat_history.pop(0)
            if len(self.chat_history) > 0:
                self.chat_history.pop(0)
            # Construct prompt again
            prompt_list = self.construct_prompt(new_prompt, user)
        return prompt_list


class Conversation:
    """
    For handling multiple conversations
    """

    def __init__(self) -> None:
        self.conversations = {}

    def add_conversation(self, key: str, history: list) -> None:
        """
        Adds a history list to the conversations dict with the id as the key
        """
        self.conversations[key] = history

    def get_conversation(self, key: str) -> list:
        """
        Retrieves the history list from the conversations dict with the id as the key
        """
        return self.conversations[key]

    def remove_conversation(self, key: str) -> None:
        """
        Removes the history list from the conversations dict with the id as the key
        """
        del self.conversations[key]

    def __str__(self) -> str:
        """
        Creates a JSON string of the conversations
        """
        return json.dumps(self.conversations)

    def save(self, file: str) -> None:
        """
        Saves the conversations to a JSON file
        """
        with open(file, "w", encoding="utf-8") as f:
            f.write(str(self))

    def load(self, file: str) -> None:
        """
        Loads the conversations from a JSON file
        """
        with open(file, encoding="utf-8") as f:
            self.conversations = json.loads(f.read())


def main():
    print(
        """
    ChatGPT - A command-line interface to OpenAI's ChatGPT (https://chat.openai.com/chat)
    Repo: github.com/acheong08/ChatGPT
    """,
    )
    print("Type '!help' to show a full list of commands")
    print("Press enter twice to submit your question.\n")

    def get_input(prompt):
        """
        Multi-line input function
        """
        # Display the prompt
        print(prompt, end="")

        # Initialize an empty list to store the input lines
        lines = []

        # Read lines of input until the user enters an empty line
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)

        # Join the lines, separated by newlines, and store the result
        user_input = "\n".join(lines)

        # Return the input
        return user_input

    def chatbot_commands(cmd: str) -> bool:
        """
        Handle chatbot commands
        """
        if cmd == "!help":
            print(
                """
            !help - Display this message
            !rollback - Rollback chat history
            !reset - Reset chat history
            !prompt - Show current prompt
            !save_c <conversation_name> - Save history to a conversation
            !load_c <conversation_name> - Load history from a conversation
            !save_f <file_name> - Save all conversations to a file
            !load_f <file_name> - Load all conversations from a file
            !exit - Quit chat
            """,
            )
        elif cmd == "!exit":
            exit()
        elif cmd == "!rollback":
            chatbot.rollback(1)
        elif cmd == "!reset":
            chatbot.reset()
        elif cmd == "!prompt":
            print(chatbot.prompt.construct_prompt(""))
        elif cmd.startswith("!save_c"):
            chatbot.save_conversation(cmd.split(" ")[1])
        elif cmd.startswith("!load_c"):
            chatbot.load_conversation(cmd.split(" ")[1])
        elif cmd.startswith("!save_f"):
            chatbot.conversations.save(cmd.split(" ")[1])
        elif cmd.startswith("!load_f"):
            chatbot.conversations.load(cmd.split(" ")[1])
        else:
            return False
        return True

    # Get API key from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream response",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for response",
    )
    args = parser.parse_args()
    # Initialize chatbot
    chatbot = Chatbot(api_key=args.api_key)
    # Start chat
    while True:
        try:
            prompt = get_input("\nUser:\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit()
        if prompt.startswith("!"):
            if chatbot_commands(prompt):
                continue
        if not args.stream:
            response = chatbot.ask(prompt, temperature=args.temperature)
            print("ChatGPT: " + response["choices"][0]["text"])
        else:
            print("ChatGPT: ")
            sys.stdout.flush()
            for response in chatbot.ask_stream(prompt, temperature=args.temperature):
                print(response, end="")
                sys.stdout.flush()
            print()


def Singleton(cls):
    instance = {}

    def _singleton_wrapper(*args, **kargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kargs)
        return instance[cls]

    return _singleton_wrapper


@Singleton
class ChatGPTBot(Bot):

    def __init__(self):
        print("create")
        self.bot = Chatbot(conf().get('open_ai_api_key'))

    def reply(self, query, context=None,conversation_id=None):
        if not context or not context.get('type') or context.get('type') == 'TEXT':
            if len(query) < 10 and "reset" in query:
                self.bot.reset()
                return "reset OK"
            return self.bot.ask(query, conversation_id=conversation_id)

