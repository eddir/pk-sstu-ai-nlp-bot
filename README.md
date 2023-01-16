# AI & NLP stend bot

This project is a bot for Telegram, which can find answers to questions based
on the information from provided sources. The bot uses an artificial neural network and NLP
techniques to find the most relevant answer.

## Installation

### Requirements

- Python 3.6+
- pip
- virtualenv
- git
- Telegram bot token
- model file (you could use the provided one)
- [navec_news_v1_1B_250K_300d_100q.tar](https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar)
- ru_small.bin (download from [here](https://github.com/bakwc/JamSpell-models/raw/master/ru.tar.gz) and unpack)

### Setup

* Clone the repository
```bash
git clone <repo_url>
```
* Create a virtual environment and activate it
```bash
virtualenv venv
source venv/bin/activate
```
* Install the requirements
```bash
pip install -r requirements.txt
```
* Create a file named `.env` in the root directory of the project and fill it with the following data:
```env
API_TOKEN=your_telegram_bot_token
```
Obtain the token from [@BotFather](https://t.me/BotFather)
* Run the bot
```bash
python bot.py
```

## Usage

The bot is able to answer questions about the following topics:
- [x] [SSTU](https://www.sstu.ru/) (information for abiturients to help them get into the university, information about the university itself)

## Machine learning

The bot uses a neural network to find the most relevant answer to the question. The model is trained on the 
[dataset.json](<>) dataset. The model is trained using the 
[Dmitry Korobchenko](https://github.com/dkorobchenko/nn-python) algorithm. You can train the model yourself using the
[training.ipynb](<>) notebook. Don't forget to provide your own dataset with most releted questions in your case.

## NLP

The bot uses [Navec](https://github.com/natasha/navec) and [Natasha](https://github.com/natasha/natasha) it self 
to get the vector representation of each word in the question and them sum them up to get the vector representation 
of the whole question. You can download the Navec model from 
[here](https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar) or train it yourself
in order to increase the accuracy (see [this](https://github.com/natasha/navec#development) for more information).

The bot uses [JamSpell](https://github.com/bakwc/JamSpell) to correct the spelling of the question. However, the
model is trained on a piece of Russian text, so it may not work well with some words. You can download the model from
[here](https://github.com/bakwc/JamSpell-models/raw/master/ru.tar.gz) or train it yourself 
(refer to [this](https://github.com/bakwc/JamSpell#train)).

## Restrictions

The bot is able to answer questions only in Russian because of the NLP models and tokenizer used in Natasha.

## License

[MIT](https://choosealicense.com/licenses/mit/)