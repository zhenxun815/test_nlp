#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Description: 
# @File: test_urllib.py
# @Project: test_nlp
# @Author: Yiheng
# @Email: GuoYiheng89@gmail.com
# @Time: 7/19/2019 14:21

from urllib import request, parse
import ssl
import requests

url = 'https://biihu.cc//account/ajax/login_process/'
dict = {'return_url': 'https://biihu.cc/',
        'user_name':  'xiaoshuaib@gmail.com',
        'password':   '123456789',
        '_post_type': 'ajax'}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/71.0.3578.98 Safari/537.36'}


def test_urllib():
    context = ssl._create_unverified_context()
    data = bytes(parse.urlencode(dict), 'utf-8')
    req = request.Request(url, data=data, headers=headers, method='POST')
    response = request.urlopen(req, context=context)
    res = response.read().decode('utf-8')
    print(res)
    print(res.encode('utf-8').decode('unicode-escape'))


def test_requests():
    res = requests.post(url, dict, headers=headers)
    print('encode {}'.format(res.encoding))
    print('content {}'.format(res.content))
    print('json {}'.format(res.json()))
    print('text {}'.format(res.text.encode('utf-8').decode('unicode-escape')))


if __name__ == '__main__':
    test_requests()
