from flask import Flask, render_template, request, redirect, send_file
from keras.models import load_model
import quandl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

WINDOW_SIZE = 245
FEATURE_LENGTH = 1


@app.route('/')
def index():
    return render_template('index.html')


def getStockDataForPrediction(ticker):
    quandl.ApiConfig.api_key = 'E38gXM5LeDsPzeKxDkY4'
    try:
        stock = quandl.get('WIKI/{}'.format(ticker))
        stock = stock.reset_index()
    except Exception as e:
        print('Error Retrieving Data.')
        print(e)

    min_date = min(stock['Date'])
    max_date = max(stock['Date'])
    print('{} Stocker initialized. Data covers {} to {}'.format(ticker, min_date.date(), max_date.date()))

    return stock


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect('/')

    future_days = int(request.form['futureDay'])
    ticker = request.form['companySelect']
    print(ticker)
    stock = getStockDataForPrediction(ticker=ticker)
    model = load_model('./saved_model/{}_lstm_model.h5'.format(ticker))

    seq_out = []
    seq_in = np.array(stock[-WINDOW_SIZE:])[:, [1]].tolist()
    today = seq_in[-1][0]

    for i in range(future_days):
        normalized_seq_in = []
        first_seq_in_value = seq_in[0][0]
        for si in range(len(seq_in)):
            normalized_seq_in.append(seq_in[si][0] / float(first_seq_in_value) - 1)
        sample_in = np.array(normalized_seq_in)
        sample_in = np.reshape(sample_in, (1, WINDOW_SIZE, FEATURE_LENGTH))
        pred_out_normalized = model.predict(sample_in)

        pred_out = first_seq_in_value * (pred_out_normalized[0][0] + 1)
        seq_out.append(pred_out)
        seq_in = seq_in[1:]
        seq_in.append([pred_out])

    model.reset_states()

    print("today's price: {:0.2f}".format(today))
    print('full prediction is : ', seq_out)
    print('{} days after today price is {:0.2f}'.format(future_days, seq_out[-1].item()))

    ### pyplot으로 그래프 저장하기
    recent_year_data = np.array(stock[stock['Date'] > (max(stock['Date']) - pd.DateOffset(years=1)).date()])[:]
    recent_year_data[:, 0]
    fig, ax = plt.subplots(1, 1)
    ax.plot(recent_year_data[:, 0], recent_year_data[:, 1], 'c', linewidth=1.4, label='Recent')
    last_date = recent_year_data[-1, 0]
    periods = future_days
    dates = pd.date_range(
        start=last_date,
        periods=periods + 1,
        freq='D'
    )
    dates = dates[dates > last_date]
    future_data_frame = pd.DataFrame({'date': dates})
    ax.plot(future_data_frame['date'], seq_out, 'm', linewidth=2.4, label='Predict')

    plt.legend(loc=2, prop={'size': 10})
    plt.xlabel('Date')
    plt.ylabel('Price $')
    plt.grid(linewidth=0.6, alpha=0.6)
    plt.title('Historical and Predicted Stock Price')
    plt.savefig("./pyplot/{}_{}.png".format(ticker, future_days))

    my_prediction = seq_out[-1]
    return render_template('results.html', prediction=my_prediction, ticker=ticker, future_days=future_days)


@app.route('/figure/<ticker>/<future_days>')
def getFigure(ticker, future_days):
    return send_file("pyplot/{}_{}.png".format(ticker, future_days), mimetype='image/gif')


if __name__ == '__main__':
    app.run(debug=True)
