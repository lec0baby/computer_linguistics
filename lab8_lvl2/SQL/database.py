import telebot
import sqlite3

bot = telebot.TeleBot('6982935311:AAGum67M_3PtOn75C_0TiYs_KctaAmyrnTs')
name = None

@bot.message_handler(commands=['start'])
def start(message):
    conn = sqlite3.connect('zador.sql')
    cur = conn.cursor()

    cur.execute('CREATE TABLE IF NOT EXISTS users (id int auto_increment primary key, username varchar(50), password varchar(50))')
    conn.commit()
    cur.close()
    conn.close()

    bot.send_message(message.chat.id, 'Приветсвтую, сейчас вас зарегистрируем! Напишите своё имя.')
    bot.register_next_step_handler(message, user_name)

def user_name(message):
    global name
    name = message.text.strip()
    bot.send_message(message.chat.id, 'Введите пароль.')
    bot.register_next_step_handler(message, user_pass)

def user_pass(message):
    password = message.text.strip()

    conn = sqlite3.connect('zador.sql')
    cur = conn.cursor()

    cur.execute('INSERT INTO users (username, password) VALUES ("%s", "%s")' % (name, password))
    conn.commit()
    cur.close()
    conn.close()

    bot.send_message(message.chat.id, 'Вы успешно зарегистрированы!')

@bot.message_handler(commands=['list'])
def list(message):
    bot.send_message(message.chat.id, 'Введите пароль высокого доступа')
    bot.register_next_step_handler(message, pass_list)

@bot.message_handler(content_types=['text'])
def pass_list(message):
    if message.text.strip() == '123zxc':
        markup = telebot.types.InlineKeyboardMarkup()
        markup.add(telebot.types.InlineKeyboardButton('Список пользователей', callback_data='users'))
        bot.send_message(message.chat.id, 'Успешный вход!', reply_markup=markup)


@bot.callback_query_handler(func=lambda call: True)
def callback(call):
    conn = sqlite3.connect('zador.sql')
    cur = conn.cursor()

    cur.execute('SELECT * FROM users')
    users = cur.fetchall()

    info = ''
    for el in users:
        info += f'Имя: {el[1]}, пароль: {el[2]}\n'

    cur.close()
    conn.close()

    bot.send_message(call.message.chat.id, info)

bot.polling(none_stop=True)