

''' Descriptive data : Non graphical - mean, SD, variance
Graphical data :  MA, MACD, Mean, basic stock price graph, trend lines,
Weighted moving average, Monthly returns for the stock '''

# Import statements
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import mean_squared_error
import math
from sklearn.linear_model import LinearRegression
import mplcursors

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

url = "https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download"
ticker_data = pd.read_csv(url)

# Making a list of tickers to check if ticker entered is listed in NASDAQ or not
ticker_list = ticker_data['Symbol'].tolist()
flag = True


# Fetching the data for the entered ticker and storing it in the data frame
def fetch_data(ticker, start, end):
    data = pdr.DataReader(ticker, "yahoo", start=start, end=end)
    if len(data) < 1:
        return False
    else:
        return data

# ----------------------------------- SUMMARY STATISTICS FOR THE GIVEN TICKER -----------------------------------


def summary_stats(df,df1):
    print('-------------------- SUMMARY STATISTICS FOR '+format(ticker)+'--------------------\n\n')
    print(df.describe())
    print('----------------------------------------------------------------------------------')
    print('\n\n-------------------- SUMMARY STATISTICS FOR CLOSING PRICES OF ' + format(ticker) + '--------------------\n\n')
    # mean of CP
    stock_mean = np.mean(df1)
    print('The average of the stock price is: ' + str(stock_mean))
    stock_min = np.min(df1)
    stock_max = np.max(df1)
    print('The range of the stock price is : {} to {}'.format(stock_min, stock_max))

    # std deviation of CP
    stock_std = np.std(df1)
    print('The standard deviation of the stock price is: ' + str(stock_std))

    # variance of CP
    stock_var = np.var(df1)
    print('The variance of the stock price is: ' + str(stock_var))

    # coefficient of variation og CP
    stock_cv = np.var(df1) / np.mean(df1)
    print('The coefficient of variation for the stock price is: ' + str(stock_cv))

# ----------------------------------- DESCRIPTIVE VISUALISATONS FOR THE GIVEN TICKER -----------------------------------


# Raw time series analysis
def time_series(df1):
    # Data set visualisations
    # Closing price raw time-series analysis
    print("RAW TIME SERIES ANALYSIS FOR " + ticker + "'s CLOSING PRICES")
    plt.figure(num="Raw time series analysis", figsize=(10, 5))
    df1.plot(grid=True, color='tab:blue')
    plt.title(ticker.upper() + "'s RAW TIME SERIES ANALYSIS")
    plt.ylabel("Closing Price($)")
    plt.xlabel("Year")
    plt.legend()
    mplcursors.cursor(hover=True)
    plt.show()
    print("-" * 100)


# Trend line
def trend_line(df1):
    # Trend line
    print("TREND LINE FOR " + ticker)
    data1 = sm.OLS(df1, sm.add_constant(range(len(df1.index)), prepend=True)).fit().fittedvalues
    plt.figure(num="Trend line "+ticker.upper()+"- closing price", figsize=(10, 5))
    df1.plot(grid=True)
    mplcursors.cursor(hover=True)
    plt.title(ticker + "'s TREND LINE - CLOSING PRICES")
    plt.plot(data1, label="trend line", color='tab:green')
    plt.ylabel("Closing price")
    plt.legend()
    plt.show()
    print("-" * 100)


# Moving average convergence/divergence
def macd(df1):
    # Moving Average Convergence Divergence
    print(ticker + "'s" + " MOVING AVERAGE CONVERGENCE DIVERGENCE")
    macd_fig = plt.figure(num="MACD",figsize=(10, 5))
    plt.grid(True)
    close_26_ewma = df1.ewm(span=26, min_periods=0, adjust=True, ignore_na=True).mean()
    close_12_ewma = df1.ewm(span=12, min_periods=0, adjust=True, ignore_na=True).mean()
    data_26 = close_26_ewma
    data_12 = close_12_ewma
    data_macd = data_12 - data_26
    plt.plot(data_26, label="EMA_26_days")
    plt.plot(data_12, label="EMA_12_days")
    plt.plot(data_macd, label="MACD")
    plt.legend(loc=2)
    plt.title(ticker + "'s MOVING AVERAGE CONVERGENCE/DIVERGENCE")
    plt.ylabel("Price($)")
    plt.xlabel("Date")
    mplcursors.cursor(hover=True)
    plt.show()
    print("-" * 100)
    return True


# Moving average / rolling mean
def rolling_mean(df,maw):
    df["Moving average"] = df1close.rolling(maw, center=True).mean()  # rolling
    plt.figure(num="Moving Average (Rolling mean)", figsize=(10, 5))
    plt.plot(df["Moving average"], label='MA ' + str(maw) + 'days')
    df1close.plot(grid=True)
    plt.legend(loc=2)
    plt.title(ticker.upper() + "'s " + str(maw) + "DAYS MOVING AVERAGE")
    plt.xlabel("Dates")
    plt.ylabel("Price($)")
    mplcursors.cursor(hover=True)
    plt.show()
    print("-" * 100)


# Monthly and daily returns visualisation for the given ticker
def returns(df1):
    monthly_returns = df1.resample('M').ffill().pct_change()
    daily_returns = df.pct_change()
    fig, ax = plt.subplots(num="Monthly and daily returns for "+ticker.upper(),nrows=2)
    ax[0].plot(monthly_returns, 'tab:blue')
    ax[1].plot(daily_returns, 'tab:green')
    ax[0].set(xlabel="Date", ylabel="Monthly returns")
    ax[1].set(xlabel="Date", ylabel="Daily returns")
    ax[0].set_title('Monthly returns')
    ax[1].set_title('Daily returns')
    mplcursors.cursor(hover=True)
    plt.tight_layout(h_pad=1.5)
    plt.show()


# Calls visualisations for the given ticker
def desc_visualisation(df1):
    ma_window = int(input("\nENTER MOVING AVERAGE WINDOW(i.e 30 = 30 days): "))  # Dynamic moving average window
    time_series(df1)
    trend_line(df1)
    rolling_mean(df, ma_window)
    bool = macd(df1)
    plt.close('all')
    if bool:
        choice = input("\nDO YOU WANT TO CALCULATE RETURNS? PLEASE ENTER Y/N: ")
        if choice == 'y' or choice == 'Y':
            returns(df1)
        elif choice == 'n' or choice == 'N':
            mmenu_return()
        else:
            print("INVALID INPUT, PLEASE ENTER A VALID OPTION")
    else:
        print(ValueError)

# ----------------------------------- PREDICTIVE ANALYTICS FOR THE GIVEN TICKER -----------------------------------


def predict_price():
    pm_strtdate = input("PLEASE ENTER THE START DATE FOR MODELLING(yyyy-mm-dd) : ")
    pm_year, pm_month, pm_day = map(int, pm_strtdate.split('-'))
    pm_start_date = dt.datetime(pm_year, pm_month, pm_day)
    pm_enddate = input("PLEASE ENTER THE END DATE FOR MODELLING(yyyy-mm-dd) : ")
    pm_year, pm_month, pm_day = map(int, pm_enddate.split('-'))
    pm_end_date = dt.datetime(pm_year, pm_month, pm_day)
    pm_data = pdr.DataReader(ticker, "yahoo", start=pm_start_date, end=pm_end_date)
    pm_data = pd.DataFrame(pm_data)
    print(pm_data.tail())

    # create train test partition
    pm_data = pm_data.reset_index()
    close = pm_data['Close'].tolist()
    dates = pm_data.index.tolist()

    # Convert to 1d Vector
    dates = np.reshape(dates, (len(dates), 1))
    prices = np.reshape(close, (len(close), 1))
    regressor = LinearRegression()
    regressor.fit(dates, prices)
    # Predicting the Test set results
    y_pred = regressor.predict(dates)
    print('Coefficients: ', regressor.coef_, '\n')
    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((regressor.predict(dates) - prices) ** 2), '\n')
    # Explained variance score: 1 is perfect prediction
    print('The coefficient of determination : %.2f' % regressor.score(dates, prices), '\n')
    mse = mean_squared_error(y_pred, prices)
    rmse = math.sqrt(mse)
    print('Root Mean square value : %.2f' % rmse, '\n')
    plt.scatter(dates, prices, color='green')  # plotting the initial datapoint
    plt.plot(dates, y_pred, color='red', linewidth=3)  # plotting the line made by linear regression
    plt.title('Linear Regression : Time vs. Price')
    plt.xlabel("No of days")
    plt.ylabel("Prices")
    plt.show()
    forecast_date = input('Enter a date in YYYY-MM-DD format for prediction :  ')
    forecast_date = (pd.to_datetime(forecast_date)).toordinal()
    today_date = pd.to_datetime(dt.datetime.now()).toordinal()
    if forecast_date >= today_date:
        nod_future = forecast_date - pm_end_date.toordinal()
        predicted_price = regressor.predict([[nod_future]])
        print("THE PREDICTED CLOSING PRICE FOR {code} IS : {predicted_price} ".format(code=ticker,
                                                                                      predicted_price=
                                                                                      predicted_price[0][0]),
              '\n')
    else:
        nod_past = today_date - forecast_date
        predicted_price1 = regressor.predict([[nod_past]])
        print("The CLOSING VALUE FOR {code} IS : {predicted_price} ".format(code=ticker,
                                                                            predicted_price=predicted_price1[0][0]),
              '\n')
    mmenu_return()


# Sub loop for returning to the main menu
def mmenu_return():
    return_choice = input("\nDO YOU WANT TO RETURN TO THE MAIN MENU? \n Please enter Y/N: ")
    if return_choice == "y" or return_choice == 'Y':
        main_menu(df, runflag=False)

    elif return_choice == "n" or return_choice == 'N':
        print("THANK YOU.")
        sys.exit()
    else:
        print("INVALID OPTION. PLEASE ENTER Y/N: ")
        mmenu_return()


def date_check():
    try:
        a=0
        strtdate_str = input('PLEASE ENTER THE START DATE FOR ANALYSIS (yyyy-mm-dd): ')
        year, month, day = map(int, strtdate_str.split('-'))
        start_date = dt.datetime(year, month, day)
        enddate_str = input('PLEASE ENTER THE END DATE FOR ANALYSIS (yyyy-mm-dd): ')
        year, month, day = map(int, enddate_str.split('-'))
        end_date = dt.datetime(year, month, day)
        strtdate = start_date.toordinal()
        enddate = end_date.toordinal()
        today = pd.to_datetime(dt.datetime.now()).toordinal()
        if strtdate == enddate:
            print('THE START AND END DATES ARE THE SAME, PLEASE ENTER THE DATES AGAIN.')
        elif enddate <= strtdate:
            print('THE START DATE HAS TO BE PRIOR TO THE END DATE, PLEASE ENTER THE DATES AGAIN.')
        elif enddate >= today:
            print('THE END DATE IS A FUTURE DATE, PLEASE ENTER THE DATES AGAIN.')
        else:
            a = 1

    except ValueError:
        print('INVALID DATES ENTERED, PLEASE ENTER THE DATES AGAIN.')
    return a, start_date, end_date

# Main user input loop
def main_menu(df, runflag, runflag1=None):
    if runflag1 != True:
        if runflag == True:
            return False
        else:
            print('-------------------- ANALYSIS FOR '+ticker+' --------------------\n\t'
                  'MENU OPTIONS\n\t'
                  '1. SUMMARY STATISTICS ON CLOSING PRICES\n\t'
                  '2. DESCRIPTIVE ANALYSIS - VISUALISATIONS \n\t'
                  '3. PREDICTIVE\n\t'
                  '4. QUIT')
            choice = int(input('\n\nPLEASE ENTER AN OPTION: '))

            if choice == 1:
                summary_stats(df, df1close)
                mmenu_return()

            elif choice == 2:
                desc_visualisation(df1close)
                mmenu_return()

            elif choice == 3:
                predict_price()

            elif choice == 4:
                print("THANK YOU FOR USING OUR STOCK ANALYSIS TOOL.")
                sys.exit()

            else:
                print("INVALID INPUT, PLEASE TRY AGAIN")
                main_menu(df, runflag=False)
    else:
        return False


runflag = False
runflag1 = False
while True:
    while True:

        try:
            ticker = "global"
            ticker = input("-------------------- STOCK_PORTFOLIO ANALYSIS --------------------"
                           "\n\nPLEASE ENTER THE COMPANY TICKER TO PERFORM ANALYSIS: ")
            if ticker.upper() not in ticker_list:
                raise ValueError
            else:
                break
        except ValueError:
            print("Invalid ticker/symbol, please enter valid ticker")

    while flag:
        try:

            a, start_date, end_date = date_check()

            if a == 1:
                df = fetch_data(ticker, start_date, end_date)
                if len(df) > 1:
                    break
        except ValueError:
            print("Exit")

    df1close = df['Close']  # Creating a data frame consisting of only the closing prices
    dfcopy = df  # Complete data set with High, Low, Close, Adj Close et al.
    data_26 = pd.DataFrame(df1close)  # For MACD - 26days
    data_12 = pd.DataFrame(df1close)  # For MACD - 12 days
    data_macd = pd.DataFrame(df1close)  # For MACD

    try:
        main_menu(df, runflag, runflag)
        if main_menu(df, runflag=False):
            break
        else:
            rerun1 = True
            raise ValueError
    except ValueError:
        print('RUNNING ANALYSIS AGAIN')
