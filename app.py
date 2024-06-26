from flask import Flask

app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Привет'

@app.route('/goodbye')
def goodbye():
    return 'До свидания'

@app.route('/goodmorning')
def goodmorning():
    return 'Доброе утро'

@app.route('/goodevening')
def goodevening():
    return 'Добрый вечер'

@app.route('/goodnight')
def goodnight():
    return 'Спокойной ночи'

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8888)
