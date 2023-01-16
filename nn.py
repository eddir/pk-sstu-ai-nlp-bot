import json

import jamspell
import numpy as np
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from navec import Navec


class Prediction:
    answers = [
        'Вопрос про бакалавриат, очная форма обучения',
        'Вопрос про целевое',
        'Вопрос про ВУЦ',
        'Вопрос про бакалавриат, заочная форма обучения',
        'Поступление в магистратуру'

        # всё что ниже предложил copilot. Можно в будущем заполнить
        # 'Вопрос про бакалавриат, очно-заочная форма обучения',
        # 'Вопрос про магистратуру, очная форма обучения',
        # 'Вопрос про магистратуру, очно-заочная форма обучения',
        # 'Вопрос про магистратуру, заочная форма обучения',
        # 'Вопрос про аспирантуру, очная форма обучения',
        # 'Вопрос про аспирантуру, очно-заочная форма обучения',
        # 'Вопрос про аспирантуру, заочная форма обучения',
        # 'Вопрос про докторантуру, очная форма обучения',
        # 'Вопрос про докторантуру, очно-заочная форма обучения',
        # 'Вопрос про докторантуру, заочная форма обучения',
        # 'Вопрос про стипендии',
        # 'Вопрос про общежития',
        # 'Вопрос про вступительные испытания',
        # 'Вопрос про документы',
        # 'Вопрос про конкурсы',
        # 'Вопрос про олимпиады',
        # 'Вопрос про сроки поступления',
        # 'Вопрос про сроки зачисления',
        # 'Вопрос про сроки перевода',
        # 'Вопрос про сроки отчисления',
        # 'Вопрос про сроки восстановления',
    ]

    def __init__(self):
        # Загружаем библиотеки Natasha и Jamspell
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.morph_vocab = MorphVocab()

        # https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar
        # https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar
        # self.navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
        self.navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')

        self.corrector = jamspell.TSpellCorrector()
        # https://github.com/bakwc/JamSpell-models/raw/master/ru.tar.gz
        self.corrector.LoadLangModel('ru_small.bin')

        with open('model.json', 'r') as f:
            model = json.load(f)

        self.W1 = np.array(model['W1'])
        self.b1 = np.array(model['b1'])
        self.W2 = np.array(model['W2'])
        self.b2 = np.array(model['b2'])

    def sentence_to_vector(self, sentence):
        sentence = sentence.replace("ВУЦ", "военный учебный центр")  # не умеем работать с аббревиатурами, увы
        sentence = sentence.replace("СГТУ", "Саратовский государственный университет")
        sentence = sentence.replace("сгту", "Саратовский государственный университет")
        text = self.corrector.FixFragment(sentence)
        doc = Doc(text)
        doc.segment(Segmenter())
        doc.tag_morph(self.morph_tagger)
        doc.tokens = [token for token in doc.tokens if
                      token.pos not in ['PUNCT', 'NUM', 'PRON', 'ADP', 'CCONJ', 'PART', 'DET', 'SCONJ']]
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        batch = [_.lemma for _ in doc.tokens]

        vectors = []
        undefined = []
        print(batch)
        for word in batch:
            vector = self.navec.get(word)
            if vector is not None:
                vectors.append(vector)
            else:
                undefined.append(word)

        if len(undefined) > 0:
            print("В корпусе не найдены некоторые слова:", undefined)

        return np.sum(vectors, axis=0)

    def run(self, inputv):
        x = np.array([self.sentence_to_vector(inputv)])
        print(f'Input: {inputv} ')

        def relu(t):
            return np.maximum(t, 0)

        def softmax(t):
            out = np.exp(t)
            return out / np.sum(out)

        def predict(x):
            t1 = x @ self.W1 + self.b1
            h1 = relu(t1)
            t2 = h1 @ self.W2 + self.b2
            z = softmax(t2)
            return z

        probs = predict(x)
        pred_class = np.argmax(probs)
        print('Predicted class: ', self.answers[pred_class])
        return self.answers[pred_class]
