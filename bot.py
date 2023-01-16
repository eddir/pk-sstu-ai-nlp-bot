import logging
import os

from aiogram import Bot, Dispatcher, executor, types

# load api token from .env file
from dotenv import load_dotenv

from nn import Prediction

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=os.getenv("API_TOKEN"))
dp = Dispatcher(bot)


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("Привет!\nЯ бот, который отвечает на вопросы о поступлении. Пиши мне!")


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(predictor.run(message.text))


if __name__ == '__main__':
    predictor = Prediction()
    executor.start_polling(dp, skip_updates=True)
