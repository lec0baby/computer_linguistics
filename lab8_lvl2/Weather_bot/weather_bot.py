import telebot
import requests
import json

bot = telebot.TeleBot('6982935311:AAGum67M_3PtOn75C_0TiYs_KctaAmyrnTs')
API = '468a74351261ce7c1bc0027ba9f33e88'

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Привет, рад тебя видеть! Напиши название города.')


@bot.message_handler(content_types=['text'])
def get_weather(message):
    city = message.text.strip().lower()
    res = requests.get(f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API}&units=metric')
    if res.status_code != 200:
        bot.reply_to(message, 'К сожалению, я не знаю такого города(')
    else:
        data = json.loads(res.text)
        temp = data["main"]["temp"]
        bot.reply_to(message, f'Сейчас погода: {temp} °C')

        image = 'sun.png' if temp > 5.0 else 'sunny.png'
        file = open(image, 'rb')
        bot.send_photo(message.chat.id, file)

bot.polling(none_stop=True)