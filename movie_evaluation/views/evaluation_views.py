import base64
import json
import os
from io import BytesIO

import nltk as nltk
import requests
from bs4 import BeautifulSoup as bs
from flask import Blueprint, render_template, request
from urllib.parse import urlparse, parse_qs, parse_qsl
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


# ----------------------- 영화 정보 파싱 메소드 -------------------------
def search_url():
    return

'''
    @method : divide_url_code
    @brief  : url주소에 code에 해당하는 값을 분리합니다.
    @detail : url주소에 쿼리스트링 code에 해당하는 값을 분리합니다.
    @author : 이성재
    @since  : 2021.09.04
    @param  : url 네이버에 영화정보가 담긴 url 주소입니다.
    @return : code값
    @why    : post의 이미지가 있는 페이지는 쿼리스트링에 code 값에 의해서 달라지기 때문에 
              같은 값을 가져와 포스터 이미지가 있는 URL을 뽑아내기 위해서
'''

# 쿼리스트링 분리
def divide_url_code(url):
    query_strings = urlparse(url)
    query_string = parse_qsl(query_strings.query)  # 쿼리스트링 파싱

    return query_string

'''
    @method : get_soup_html
    @brief  : 해당 url을 참조하는 beautifulSoup 객체를 만듭니다.
    @detail : 네이버 영화정보 담긴 url을 참조하는 beautifulSoup 객체를 만듭니다.
    @author : 이성재
    @since  : 2021.09.04
    @param  : url 네이버 영화정보 담긴 url입니다.
    @return : 네이버 영화 정보 담긴 url을 참조하는 beautifulSoup 객체
    @why    : 정적크롤링을 이용해 영화 정보를 가져오기 위해서 만들었습니다.
'''

# BeautifulSoup 객체생성
def get_soup_html(url):
    html = requests.get(url)
    soup = bs(html.text, 'html.parser')

    return soup

'''
    @method : get_movie_name
    @brief  : 영화제목을 가져오는 함수입니다.
    @detail : 영화제목을 가져오는 함수입니다.
    @author : 이성재
    @since  : 2021.09.05
    @param  : url 네이버 영화정보 담긴 url입니다.
    @return : 네이버 영화 정보 담긴 페이지에서 영화 제목을 리턴합니다.
    @why    : 영화 뉴스 기사 빈도분석을 위해 영화 제목이 필요해 만들었습니다.
'''

# 제목
def get_movie_name(soup):
    h_movie = soup.find(class_="h_movie")
    movie_name = h_movie.find("a").text

    return movie_name

'''
    @method : get_star_ratings
    @brief  : 관람객, 네티즌, 기자 평론가 평점을 가져오는 함수입니다.
    @detail : 영화의 관람객, 네티즌, 기자 평론가 평점을 가져오는 함수입니다.
    @author : 이성재
    @since  : 2021.09.04
    @param  : soup 네이버 영화페이지 url을 참조하는 BeautifulSopu 객체입니다.
    @return : 관람객, 네티즌, 기자 평론가 평점이 리스트로 담겨 반환됩니다.
    @why    : 관람객, 네티즌, 기자 평론가 평점을 가져오기 위해서 만들었습니다.
'''

# 관람객, 네티즌, 기자 평론가 평점
def get_star_ratings(soup):
    main_score = soup.find(class_="main_score")
    star_scores = main_score.findAll(class_="star_score")
    star_rating = ""
    star_ratings = []

    for star_score in star_scores:
        rating_letters = star_score.findAll("em")
        for rating_letter in rating_letters:
            star_rating += rating_letter.text  # 숫자가 조각나 있어서 합치기 위해 이러한 행위를 한다.
        star_ratings.append(star_rating)
        star_rating = ""

    return star_ratings

'''
    @method : get_infos
    @brief  : 개요, 감독, 배우, 등급, 흥행을 가져오는 함수입니다.
    @detail : 영화의 개요, 감독, 배우, 등급, 흥행을 가져오는 함수입니다.
    @author : 이성재
    @since  : 2021.09.04
    @param  : soup 네이버 영화페이지 url을 참조하는 BeautifulSopu 객체입니다.
    @return : 개요, 감독, 배우, 등급, 흥행이 리스트로 담겨 반환됩니다.
    @why    : 개요, 감독, 배우, 등급, 흥행을 가져오기 위해서 만들었습니다.
'''

# 개요, 감독, 배우, 등급, 흥행
def get_infos(soup):
    info_spec = soup.find("dl", {"class": "info_spec"})
    info_specs = info_spec.findAll("dd")
    infos = []
    infos_list = []

    for info in info_specs:
        info_contents = info.findAll("a")
        for info_content in info_contents:
            infos.append(info_content.text)
        infos_list.append(infos)
        infos = []

    return infos_list

'''
    @method : get_poster_url
    @brief  : 포스터 이미지 정보가 있는 url을 가지고 img태그의 src값을 가져옵니다.
    @detail : code 쿼리스트링을 추출한 함수로 code를 받아 포스터 이미지 정보가 있는 url을 만들어 src의 주소값을 가져옵니다.
    @author : 이성재
    @since  : 2021.09.04
    @param  : code url에 있는 영화에 해당하는 코드번호입니다.
    @return : src속성의 값을 반환합니다.
    @why    : src속성의 값을 가져와 이용하기 위해서 만들었습니다.
'''


# 포스터 img src의 주소값 구하기
def get_poster_url(code):
    post_url = 'https://movie.naver.com/movie/bi/mi/photoViewPopup.naver?movieCode=' + code

    html = requests.get(post_url)
    soup = bs(html.text, 'html.parser')
    post_img_tag = soup.find('img', id='targetImage')
    post_img_url = post_img_tag.attrs['src']

    return post_img_url


'''
    @method : get_comment_scores
    @brief  : 댓글의 평점을 가져옵니다.
    @detail : 1페이지에 있는 댓글의 평점을 가져옵니다.
    @author : 이성재
    @since  : 2021.09.04
    @param  : soup 네이버 영화페이지 url을 참조하는 BeautifulSopu 객체입니다.
    @return : 댓글 평점이 담긴 list를 반환합니다.
    @why    : 댓글의 평점을 가져오기 위해 만들었습니다.
'''


# 댓글 평점 가져오기
def get_comment_scores(soup):
    comment_score_list = []
    score_results = soup.findAll(class_="score_result")

    for score_result in score_results:
        star_scores = score_result.findAll(class_="star_score")
        for star_score in star_scores:
            comment_score_list.append(star_score.find("em").text)

    return comment_score_list


'''
    @method : get_comments
    @brief  : 댓글들을 가져옵니다.
    @detail : 1페이지에 있는 댓글들을 가져옵니다.
    @author : 이성재
    @since  : 2021.09.04
    @param  : soup 네이버 영화페이지 url을 참조하는 BeautifulSopu 객체입니다.
    @return : 댓글들이 담긴 list를 반환합니다.
    @why    : 댓글들을 가져오기 위해 만들었습니다.
'''

# 댓글 가져오기
def get_comments(soup):
    comment_list = []
    score_reples = soup.findAll(class_="score_reple")
    for score_reple in score_reples:
        comment_list.append(score_reple.find("p").text.strip())

    return comment_list

'''
    @method : get_sympathys
    @brief  : 각 댓글의 추천수를 가져옵니다.
    @detail : 1페이지에 있는 각 댓글의 추천수를 가져옵니다..
    @author : 이성재
    @since  : 2021.09.04
    @param  : soup 네이버 영화페이지 url을 참조하는 BeautifulSopu 객체입니다.
    @return : 각 댓글의 추천수가 담긴 list를 반환합니다.
    @why    : 각 댓글의 추천수를 가져오기 위해 만들었습니다.
'''

# 댓글의 추천수
def get_sympathys(soup):
    sympathy_list = []
    sympathy_buttons = soup.findAll(class_="_sympathyButton")

    for sympathy_button in sympathy_buttons:
        sympathy_list.append(sympathy_button.find("strong").text)

    return sympathy_list

'''
    @method : get_not_sympathys
    @brief  : 각 댓글의 비추천수를 가져옵니다.
    @detail : 1페이지에 있는 각 댓글의 비추천수를 가져옵니다..
    @author : 이성재
    @since  : 2021.09.04
    @param  : soup 네이버 영화페이지 url을 참조하는 BeautifulSopu 객체입니다.
    @return : 각 댓글의 비추천수가 담긴 list를 반환합니다.
    @why    : 각 댓글의 비추천수를 가져오기 위해 만들었습니다.
'''

# 댓글의 비추수
def get_not_sympathys(soup):
    not_sympathy_list = []
    not_sympathy_buttons = soup.findAll(class_="_notSympathyButton")
    for not_sympathy_button in not_sympathy_buttons:
        not_sympathy_list.append(not_sympathy_button.find("strong").text)

    return not_sympathy_list

# ----------------------- 빈도분석 함수 ------------------------------

'''
    @method : get_titles
    @brief  : 영화 뉴스 기사 제목들을 크롤링합니다.
    @detail : 네이버 뉴스에 영화 제목을 검색해 뉴스 기사 제목들을 크롤링합니다.
    @author : 이성재
    @since  : 2021.09.06
    @param  : search_word 검색 키워드,
              start_num   뉴스 목록 시작페이지,
              end_num     뉴스 목록 끝 페이지,
    @return : 크롤링한 뉴스 제목 리스트를 반환합니다.
    @why    : 뉴스 제목의 단어 빈도분석을 하기 위해 만들었습니다.
'''

# 네이버 뉴스에 영화 제목을 검색하는 크롤링
def get_titles(search_word, start_num , end_num):
    title_list = []

    # 시작 페이지(start_num) ~ 끝 페이지(end_num) 까지 반복
    while 1:
        if start_num > end_num:
            return title_list
            break

        url = 'https://search.naver.com/search.naver?where=news&sm=tab_jum&query={}&start={}'\
            .format(search_word,start_num)

        req = requests.get(url)

        # 주소를 제대로 가져왔을 시 파싱 시작
        if req.ok :
            html = req.text
            soup = bs(html, 'html.parser')

            # 뉴스제목 뽑아오기
            titles = soup.select('.news_area a[title]')

            # 뉴스제목들 list에 저장
            for title in titles:
                title_list.append(title.text)

        start_num += 10

'''
    @method : sentence_tag
    @brief  : 문장을 형태소로 분류합니다.
    @detail : 영화 뉴스 제목들이 담긴 리스트들을 (단어 : 형태소) 형태로 분류합니다.
    @author : 이성재
    @since  : 2021.09.06
    @param  : title_list 영화 뉴스 제목가 담긴 리스트입니다.
    @return : 영화 뉴스 제목을 형태소 분석해 (단어 : 형태소)로 반환합니다.
    @why    : 형태소로 분석해 명사와 형용사 부분만 가져오기 위해서 필요한 사전작업이기 때문에 만들었습니다.
'''

# 문장을 형태소로 분류
def sentence_tag(title_list):
    okt = Okt()
    sentences_tag = []

    # 형태소 분석하여 리스트에 저장
    for sentence in title_list :
        morph = okt.pos(sentence)
        sentences_tag.append(morph)

    return sentences_tag

'''
    @method : word_count
    @brief  : 명사와 형용사를 구분해서 같은 단어 갯수를 카운트하는 함수입니다.
    @detail : 명사와 형용사를 고르고 같은 단어 갯수를 카운트합니다.
    @author : 이성재
    @since  : 2021.09.06
    @param  : sentences_tag (단어 : 형태소) 형태로 이루어진 리스트입니다.
    @return : (단어 : 카운트 수) 형태로 반환합니다.
    @why    : 카운트 수가 많은 단어일 수록 더 크게 보여져야 하기 때문에 각 단어들을 카운트합니다.
'''

# 단어별 카운트
def word_count(sentences_tag):
    noun_adj_list = []

    # 명사와 형용사만 구분하여 noun_adj_list에 저장
    for sentence in sentences_tag :
        for word, tag in sentence:
            if tag in ['Noun' , 'Adjective']:
                noun_adj_list.append(word)

    # 형태소별 갯수 count
    counts = Counter(noun_adj_list)
    tags = counts.most_common()

    return tags

'''
    @method : wordCloud_Image
    @brief  : 카운트된 데이터를 이미지화 합니다.
    @detail : 카운트된 데이터를 이미지화 합니다.
    @author : 이성재
    @since  : 2021.09.06
    @param  : word_count_list (단어 : 단어갯수)
    @return : 만든 클라우드 이미지를 png 형태로 저장해 byte형태로 바꿔서 반환합니다.
    @why    : 클라우드 이미지를 만들어 라우트의 값을 넘길 때 그냥 넘길 수 없기 때문에 byte로 바꿔서 보내 다시 이미지화 하기 위해 만들었습니다.
'''

# 단어별로 카운트된 데이터를 이미지화
def wordCloud_Image(word_count_list):

    # wordCloud 생성 (한글 폰트중 일부는 지원 않되는 것이 있음)
    wc = WordCloud(font_path="C:/Winodws/Fonts/batang.ttc", background_color = 'white', width = 800, height = 600)
    counts = (dict(word_count_list))

    cloud = wc.generate_from_frequencies(counts)
    img = BytesIO()

    plt.figure(figsize=(10,8))
    plt.axis('off')
    plt.imshow(cloud)
    plt.savefig(img, format='png')
    img.seek(0)

    # png 형태 이미지를 byte로 바꿔서 라우트 함수로 보내 HTML에 사용할 수 있게 한다.
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


#------------------------------- 영화 리뷰 평가 --------------------------------

# 형태소를 분리에 필요한 클래스를 만듭니다.
okt = Okt()

# 긍정, 부정 분석을 학습한 모델 결과를 불러옵니다.
file_name = os.path.dirname(__file__) + '/naver_data/msmc_model.h5'
model = load_model(file_name)

# 어떤 단어가 어떤 형태소인지에 대해 적힌 json 파일을 이용해 빈도 분석을 합니다.
file_json = os.path.dirname(__file__) + '/naver_data/train_docs.json'
with open(file_json, encoding="cp949") as f:
    train_docs = json.load(f)

tokens = [t for d in train_docs for t in d[0]]
text = nltk.Text(tokens, name='NMSC')

# 출현 빈도가 높은 상위 토큰 2000개을 가져와 리스트 형식으로 넣는다
selected_words = [f[0] for f in text.vocab().most_common(2000)]

'''
    @method : tokenize
    @brief  : 문장을 (단어/형태소) 형태로 분류해 리스트로 만듭니다.
    @detail : 문장을 (단어/형태소) 형태로 분류해 리스트로 만듭니다.
    @author : 이성재
    @since  : 2021.09.06
    @param  : doc 분석할 문장입니다.
    @return : 문장을 (단어/형태소)형태로 분류해 리스트로 반환합니다.
    @why    : 문장을 긍정인지 부정인지 분석하기 위해서 형태소 단위로 쪼개야 긍정, 부정이 분류되어 있는 파일과 분류해 긍정인지 부정인지 알 수 있기 때문에 만들었습니다.
'''

# 문장을 (단어/형태소) 형태로 분류해 리스트로 만듭니다.
def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

'''
    @method : term_frequency
    @brief  : 단어/형태소 형태로 분류한 거
    @detail : 출현 빈도가 높은 2000개의 단어에 해당하는 것이 있는지 확인하고 몇번 나오는지 갯수를 세어준다.
    @author : 이성재
    @since  : 2021.09.06
    @param  : doc (단어/형태소) 형태한 리스트입니다.
    @return : 2000개의 단어 데이터를 가지고 리스트로 만듭니다.
    @why    : 문장을 긍정인지 부정인지 분석하기 위해서 형태소 단위로 쪼개야 긍정, 부정이 분류되어 있는 파일과 분류해 긍정인지 부정인지 알 수 있기 때문에 만들었습니다.
'''

# 출현 빈도가 높은 2000개의 단어에 해당하는 것이 있는지 확인하고 몇번 나오는지 갯수를 세어준다.
def term_frequency(doc):
    return [doc.count(word) for word in selected_words]  # 2000개 중에 포함되는 단어

'''
    @method : predict_pos_neg
    @brief  : 문장을 (단어/형태소) 형태로 분류해 리스트로 만듭니다.
    @detail : 문장을 (단어/형태소) 형태로 분류해 리스트로 만듭니다.
    @author : 이성재
    @since  : 2021.09.06
    @param  : doc 분석할 문장입니다.
    @return : 문장을 (단어/형태소)형태로 분류해 리스트로 반환합니다.
    @why    : 문장을 긍정인지 부정인지 분석하기 위해서 형태소 단위로 쪼개야 긍정, 부정이 분류되어 있는 파일과 분류해 긍정인지 부정인지 알 수 있기 때문에 만들었습니다.
'''

def predict_pos_neg(review):
    # 리뷰 내용 형태소 분석합니다.
    token = tokenize(review)

    # 2000개의 단어에 몇개가 해당하는지 카운트 합니다.
    tf = term_frequency(token)

    # 컴퓨터가 읽을 수 있게 변환합니다.
    data = np.expand_dims(np.asarray(tf).astype('float32'), axis=0)

    # 컴퓨터가 확률로 점수를 줍니다.
    score = float(model.predict(data))

    positive = "Best 댓글들은 {:.2f}% 확률로 긍정 리뷰입니다. 추천드립니다. \n".format(score * 100)
    negative = "Best 댓글들은 {:.2f}% 확률로 부정 리뷰입니다. 비추천드립니다. \n".format((1 - score) * 100)

    comment_evaluation_list = []

    if (score > 0.5):
        is_review_evaluation = "positive"

        comment_evaluation_list.append(positive)
        comment_evaluation_list.append(is_review_evaluation)

        return comment_evaluation_list
    else:
        is_review_evaluation = "negative"

        comment_evaluation_list.append(negative)
        comment_evaluation_list.append(is_review_evaluation)
        return comment_evaluation_list


# ------------------------------------------------------------------

# ----------------------------- bp ---------------------------------

bp = Blueprint('evaluation',__name__, url_prefix='/evaluation')


@bp.route('/', methods=('GET','POST'))
def page():

    if request.method == 'POST':
        # 입력받은 url 가져오기
        movie_url = request.form['url']

        # soup 객체 만들기
        soup = get_soup_html(movie_url)

        # 쿼리스트링 code 값 가져오기
        query_strings = divide_url_code(movie_url)
        code = query_strings[0][1]

        # 제목 가져오기
        movie_name = get_movie_name(soup)

        # 관람객, 네티즌, 기자 평론가 평점가져오기
        star_ratings = get_star_ratings(soup)

        # 개요, 감독, 배우, 등급, 흥행가져오기
        infos = get_infos(soup)

        # 포스터 가져오기
        img = get_poster_url(code)

        # 댓글평점, 댓글, 추천수, 비추천수 가져오기
        comment_scores = get_comment_scores(soup)
        comments = get_comments(soup)
        sympathys = get_sympathys(soup)
        not_sympathys = get_not_sympathys(soup)

        # 뉴스기사로 빈도 분석하기
        title_list = get_titles(movie_name, 1, 100)
        sentences_tag = sentence_tag(title_list)
        word_count_list = word_count(sentences_tag)
        img_plot_url = wordCloud_Image(word_count_list)

        # (Best 댓글) 영화 리뷰들 평가하기
        comment_evaluation_list = predict_pos_neg(str(comments))

        return render_template('evaluation/evaluation.html',
                               star_ratings=star_ratings,
                               infos=infos, img=img,
                               comment_scores=comment_scores, comments=comments, sympathys=sympathys, not_sympathys=not_sympathys, img_plot_url=img_plot_url, comment_evaluation_list=comment_evaluation_list)

    return render_template('evaluation/evaluation.html')

@bp.route('/manual', methods=('GET','POST'))
def manual():
    return render_template('evaluation/manual.html')
