import requests
import re
import string
import os
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

# letters = list(string.ascii_uppercase)
# href_list = []
#
# for letter in tqdm(letters, total=len(letters)):
#
#     disease_dic = {}
#     r = requests.get('http://www.mayoclinic.org/diseases-conditions/index?letter={}'.format(letter),
#                      headers = {'Host':'www.mayoclinic.org','User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:64.0) Gecko/20100101 Firefox/64.0'}).text
#
#     soup = bs(r, "html.parser")
#     try:
#         items = soup.find("div", id="index").find("ol").find_all("li")
#     except AttributeError:
#         print(letters)
#         #items = soup.find("div", id="index").find("ol").find_all("li")
#
#     for item in items:
#         # item_ascii = re.sub(u"\u2018|\u2019", "'", item.text).replace(u"\u2014", "-")
#         # print(item.span.text.split('(See')[0].strip())
#         href_list.append('http://www.mayoclinic.org' + item.a['href'])
#
# href_list = set(href_list)
# with open('disease_urls.txt', 'w') as fout:
#     for url in href_list:
#         fout.write(url + '\n')

disease_dic = {}
with open('disease_urls.txt', 'r') as fin:
    urls = [line.strip() for line in fin]
    for i in tqdm(range(0, len(urls))):
        req = requests.get(urls[i],
                           headers={'authority': 'www.mayoclinic.org', 'user-agent': 'Mozilla/5.0 (Linux; Android '
                                                                                     '6.0; Nexus 5 Build/MRA58N) '
                                                                                     'AppleWebKit/537.36 (KHTML, '
                                                                                     'like Gecko) '
                                                                                     'Chrome/85.0.4183.121 Mobile '
                                                                                     'Safari/537.36'}).text
        sp = bs(req, "html.parser")
        title = sp.find("div", class_="main").header.find("h1").text
        content = sp.find("article", id="main-content").find("div", class_="content").find("div", class_=None)
        to_del = content.find_all("div")
        for div in to_del:
            div.extract()
        with open('./disease_condition/' + title.replace('/', '／').replace('\\', '＼') + '.txt', 'w',
                  encoding='utf-8') as fout:
            fout.write(str(content))
