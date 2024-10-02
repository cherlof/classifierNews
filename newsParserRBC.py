import requests
from bs4 import BeautifulSoup
import json


def get_news():
    url = "https://sportrbc.ru/football/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    articles = []
    for article in soup.select('a.item__link.rm-cm-item-link')[:20]:
        title_tag = article.select_one('span.item__title.rm-cm-item-text')
        link = article['href']

        if title_tag and link:
            title = title_tag.get_text(strip=True)
            print(f"Processing article: {title}")
            article_response = requests.get(link)
            article_soup = BeautifulSoup(article_response.content, 'html.parser')
            text_tag = article_soup.select_one('div.article__text')

            if text_tag:
                text = text_tag.get_text(strip=True)
                articles.append({
                    'title': title,
                    'text': text
                })
            else:
                print(f"Text not found for article: {title}")
        else:
            print("Title or link not found for an article.")

    if articles:
        with open('news.json', 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        print("Articles saved to news.json")
    else:
        print("No articles found.")


if __name__ == "__main__":
    get_news()