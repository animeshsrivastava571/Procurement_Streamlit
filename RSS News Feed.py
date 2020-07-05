from bs4 import BeautifulSoup
import requests
import pandas as pd


def news_covid_RSS(url):
    resp = requests.get(url)
    if (resp.status_code==200):
        resp_text = resp.text
        soup = BeautifulSoup(resp_text,'xml')
        items = soup.findAll('item')
        lst_link = []
        lst_des=[]
        for item in items:
            str1=item.description.text
            soup1 = BeautifulSoup(str1,'xml')
            lst_link.append(soup1.a['href'])
            lst_des.append(soup1.text)
        df = pd.DataFrame({'news_link':lst_link,'news_feed':lst_des})
        df.to_csv('News_Feed_Corona.csv',index=None)
        print(df)
    return 0
            



if __name__ == "__main__":
    url = 'http://news.google.com/news?q=covid-19&hl=en-US&sort=date&gl=US&num=100&output=rss'
    news_covid_RSS(url)
    

