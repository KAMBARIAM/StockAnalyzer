import sys
import pandas as pd
from nsepy import symbols
from GoogleNews import GoogleNews
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_stock_news(company_name):
    googlenews = GoogleNews(lang='en', period='1d')
    googlenews.search(company_name)
    #result1 = googlenews.page_at(1)
    #result2 = googlenews.page_at(2)
    results = googlenews.results(sort=True) #result1 + result2
    googlenews.clear()
    return results

def get_comp_name(stock_symbol):
    symbols_df = symbols.get_symbol_list()
    company_name = symbols_df[symbols_df["SYMBOL"] == stock_symbol]["NAME OF COMPANY"].values[0]
    return company_name

def analyze_and_create_file(news, news_columns):
    vader = SentimentIntensityAnalyzer()
    parsed_and_scored_news = pd.DataFrame(news, columns=news_columns)
    scores = parsed_and_scored_news['desc'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    parsed_and_scored_news.to_csv(sys.argv[1] + "_sentiment.csv", sep='\t', encoding='utf-8')
    print("File",sys.argv[1] + "_sentiment.csv is created")
    return parsed_and_scored_news

def prepare_news(results):
    news = []
    news_columns = ['datetime','date','title', 'desc', 'link']
    for news_item in results:
        news.append([news_item['datetime'], news_item['date'], news_item['title'], news_item['desc'], news_item['link']])
    return (news, news_columns)

def print_news():
    company_name = sys.argv[1] #get_comp_name(stock_symbol = sys.argv[1])
    results = get_stock_news(company_name=company_name)
    news, news_columns = prepare_news(results=results)
    parsed_and_scored_news = analyze_and_create_file(news=news, news_columns=news_columns)
    print(parsed_and_scored_news)

if __name__ == '__main__':
    print_news()