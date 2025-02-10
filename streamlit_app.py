




import streamlit as st
import colorsys
import requests
import pandas as pd
import re
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import spacy
import nltk
from nltk.util import ngrams  # for bigrams/trigrams

# For topic modelling
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

# For interactive network visualisation
from pyvis.network import Network
import streamlit.components.v1 as components

# For TF-IDF, clustering, LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

# For ChatGPT calls
import openai

# Load spaCy
nlp = spacy.load("en_core_web_sm")


# 1) Helper Functions
def validate_api_key(api_key):
    url = 'https://newsapi.org/v2/top-headlines'
    params = {'country': 'us', 'pageSize': 1}
    headers = {'Authorization': f'Bearer {api_key}'}
    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()

    if resp.status_code == 200 and data.get('status') == 'ok':
        return True
    else:
        msg = data.get('message', 'Unknown error')
        raise ValueError(f"Invalid API key or error from NewsAPI: {msg}")


def fetch_articles(api_key, query, language='en', sort_by='publishedAt', from_date=None, to_date=None):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'language': language,
        'sortBy': sort_by
    }
    if from_date:
        params['from'] = from_date
    if to_date:
        params['to'] = to_date

    headers = {'Authorization': f'Bearer {api_key}'}
    resp = requests.get(url, headers=headers, params=params)
    data = resp.json()

    if resp.status_code != 200 or data.get('status') != 'ok':
        raise ValueError(f"NewsAPI error: {data.get('message', 'Unknown error')}")
    return data['articles']


def clean_text_more_thoroughly(title, description, content):
    combined = f"{title or ''} {description or ''} {content or ''}".lower()
    combined = re.sub(r'\d+', '', combined)
    combined = re.sub(r'[^\w\s]', '', combined)
    combined = re.sub(r'\s+', ' ', combined)
    combined = combined.strip()
    return combined


def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment


def compute_word_frequency(text_series):
    freq = {}
    for txt in text_series:
        for w in txt.split():
            freq[w] = freq.get(w, 0) + 1
    return freq


def create_wordcloud(all_text, stopwords=None):
    if stopwords is None:
        stopwords = set()
    wc = WordCloud(width=600, height=400, background_color='white', stopwords=stopwords).generate(all_text)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig


def apply_stopwords_union(text, custom_stopwords):
    combined_stopwords = STOPWORDS.union(custom_stopwords)
    tokens = text.split()
    filtered = [w for w in tokens if w not in combined_stopwords]
    return ' '.join(filtered)


def lemmatise_text_spacy(txt):
    doc = nlp(txt)
    lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    return " ".join(lemmas)


def generate_ngrams(txt, n=2):
    tokens = txt.split()
    ngram_tuples = list(ngrams(tokens, n))
    ngram_strings = ["_".join(pair) for pair in ngram_tuples]
    return " ".join(ngram_strings)


def extract_entities_spacy(title, description, content):
    raw_text = f"{title or ''} {description or ''} {content or ''}"
    doc = nlp(raw_text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# Gensim LDA
def run_lda_topic_model(docs, num_topics=5, passes=10, num_words=10):
    tokens_list = [doc.split() for doc in docs if doc.strip()]
    dictionary = Dictionary(tokens_list)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
    return lda_model, corpus, dictionary


def create_topic_pyvis_network(topic_id, topic_terms):
    net = Network(height="600px", width="100%", directed=False)
    net.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 16, "align": "center"},
        "shape": "circle"
      },
      "edges": {
        "smooth": false,
        "color": {"inherit": false}
      },
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 100}
      },
      "interaction": {"dragNodes": true}
    }
    """)

    center_node_id = f"Topic_{topic_id}"
    net.add_node(center_node_id, label=f"Topic {topic_id}", size=25, color="#ff7f0e")

    for (term, weight) in topic_terms:
        weight_val = float(weight)
        size = 10 + (weight_val * 3000.0)
        net.add_node(term, label=term, size=size, color="#1f77b4")
        net.add_edge(center_node_id, term, value=weight_val, title=f"Weight: {weight_val:.5f}")
    return net


def display_pyvis_network(net, topic_id):
    html_filename = f"topic_network_{topic_id}.html"
    net.write_html(html_filename)
    with open(html_filename, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600, scrolling=False)


def extract_keywords_tfidf(docs, top_n=10):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    avg_tfidf = np.mean(X.toarray(), axis=0)
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = np.argsort(avg_tfidf)[::-1]
    top_indices = sorted_indices[:top_n]
    top_keywords = [(feature_names[i], avg_tfidf[i]) for i in top_indices]
    return top_keywords


# def cluster_documents_kmeans(docs, num_clusters=5):
#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(docs)
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(X)
#     labels = kmeans.labels_
#     # feature_names = vectorizer.get_feature_names_out()
#     # return labels, kmeans, vectorizer, X
#     return kmeans, vectorizer, X


def cluster_documents_kmeans(docs, num_clusters=5):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Return the model, the vectorizer object, and X if needed
    return kmeans, vectorizer, X, labels







# def get_top_terms_per_cluster(kmeans, vectorizer, n_top_words=10):
#     centroids = kmeans.cluster_centers_
#     feature_names = vectorizer.get_feature_names_out()
#     results = {}

#     for cluster_id, centroid in enumerate(centroids):
#         sorted_indices = np.argsort(centroid)[::-1]
#         top_features = [
#             (feature_names[i], centroid[i]) 
#             for i in sorted_indices[:n_top_words]
#         ]
#         results[cluster_id] = top_features
    
#     return results







def get_top_terms_per_cluster(kmeans, vectorizer, num_terms=10):
    centroids = kmeans.cluster_centers_
    # Now we call the vectorizer method to get real feature names
    feature_names = vectorizer.get_feature_names_out()

    results = {}
    for cluster_id, centroid in enumerate(centroids):
        sorted_indices = np.argsort(centroid)[::-1]
        top_features = [
            (feature_names[i], centroid[i])
            for i in sorted_indices[:num_terms]
        ]
        results[cluster_id] = top_features
    return results








# Additional scikit-based LDA code
def display_top_words_for_lda(lda_model, feature_names, n_top_words=10):
    results = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        results[topic_idx] = top_words
    return results

def run_sklearn_lda_topic_modelling(docs, n_topics=5, n_top_words=10, max_iter=10):
    vectorizer = CountVectorizer(stop_words=None)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, random_state=42)
    lda.fit(X)
    doc_topic_matrix = lda.transform(X)
    return lda, doc_topic_matrix, feature_names

def run_kmeans_clustering_sklearn(docs, n_clusters=5, n_top_words=10):
    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    return kmeans, labels, feature_names, X


# 2) Streamlit Main App
def main():
    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #006400 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] .css-1d391kg {
        padding-top: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("News Analysis Extended + Detailed LLM Narratives")

    if 'api_key_validated' not in st.session_state:
        st.session_state.api_key_validated = False
    if 'articles_df' not in st.session_state:
        st.session_state.articles_df = pd.DataFrame()
    if 'custom_stopwords' not in st.session_state:
        st.session_state.custom_stopwords = set()

    # We'll track the LLM key as well
    if 'llm_api_key' not in st.session_state:
        st.session_state.llm_api_key = None
    if 'llm_key_validated' not in st.session_state:
        st.session_state.llm_key_validated = False

    # We'll also store the "Detailed Topics & Clusters" results so that
    # the new "Narratives" tab can access them.
    if 'df_for_thematic' not in st.session_state:
        st.session_state.df_for_thematic = pd.DataFrame()
    if 'lda_topic_top_words' not in st.session_state:
        st.session_state.lda_topic_top_words = {}
    if 'cluster_top_terms' not in st.session_state:
        st.session_state.cluster_top_terms = {}
    if 'topic_assignments' not in st.session_state:
        st.session_state.topic_assignments = []
    if 'cluster_assignments' not in st.session_state:
        st.session_state.cluster_assignments = []

    # SIDEBAR
    st.sidebar.markdown("# News Analysis\n---")

    if st.sidebar.button("Reset Data & Analyses"):
        st.session_state.clear()
        st.experimental_rerun()

    st.sidebar.header("NewsAPI Settings")
    news_api_key = st.sidebar.text_input("Enter your NewsAPI key", type="password", value="YOUR_NEWS_API_KEY_HERE")
    if st.sidebar.button("Validate NewsAPI Key"):
        if not news_api_key:
            st.sidebar.error("Please provide a NewsAPI key.")
        else:
            with st.spinner("Validating NewsAPI key..."):
                try:
                    validate_api_key(news_api_key)
                    st.session_state.api_key_validated = True
                    st.sidebar.success("NewsAPI key is valid!")
                except Exception as e:
                    st.session_state.api_key_validated = False
                    st.sidebar.error(f"Key invalid or error occurred: {e}")

    st.sidebar.markdown("---")

    # ChatGPT Key
    st.sidebar.header("LLM Settings (ChatGPT)")
    llm_key = st.sidebar.text_input("Enter your ChatGPT API key", type="password")
    if st.sidebar.button("Validate ChatGPT Key"):
        if not llm_key:
            st.sidebar.error("Please provide a ChatGPT API key.")
        else:
            st.session_state.llm_api_key = llm_key
            st.session_state.llm_key_validated = True
            st.sidebar.success("LLM key stored. Ready for narratives!")

    st.sidebar.markdown("---")

    # Search Parameters
    query = st.sidebar.text_input("Search Query", value="Python")
    language = st.sidebar.selectbox("Language", ["en", "es", "fr", "de"])
    sort_by = st.sidebar.selectbox("Sort By", ["publishedAt", "relevancy", "popularity"])
    enable_date_filter = st.sidebar.checkbox("Filter by a date range?", value=False)
    if enable_date_filter:
        from_date = st.sidebar.date_input("From Date")
        to_date = st.sidebar.date_input("To Date")
    else:
        from_date = None
        to_date = None

    # Fetch
    if st.sidebar.button("Fetch News"):
        if not news_api_key:
            st.error("Please provide a NewsAPI key.")
            return
        if not st.session_state.api_key_validated:
            st.error("Your NewsAPI key is not validated. Please validate it before fetching.")
            return
        from_date_str = from_date.isoformat() if from_date else None
        to_date_str = to_date.isoformat() if to_date else None
        with st.spinner("Fetching articles..."):
            try:
                articles = fetch_articles(news_api_key, query, language, sort_by, from_date_str, to_date_str)
                if not articles:
                    st.warning("No articles found. Try a different query or date range.")
                else:
                    df = pd.DataFrame(articles)
                    df['publication'] = df['source'].apply(lambda s: s.get('name', 'Unknown'))
                    df['cleanedText'] = df.apply(
                        lambda row: clean_text_more_thoroughly(row.title, row.description, row.content),
                        axis=1
                    )
                    st.session_state.articles_df = df
                    st.success(f"Fetched {len(df)} articles.")
            except Exception as e:
                st.error(f"Error fetching or processing results: {e}")

    df = st.session_state.articles_df
    if df.empty:
        st.info("No articles fetched yet. Please fetch news to proceed.")
        return

    # TABS
    tab_stopwords, tab_ner, tab_topics, tab_keywords, tab_clustering, tab_sentviz, tab_topics_clusters, tab_narratives = st.tabs([
        "Stopwords & Advanced", 
        "NER Tab", 
        "Topic Modelling",
        "Keyword Extraction", 
        "Clustering & Classification",
        "Sentiment Visualisation",
        "Detailed Topics & Clusters",
        "Narratives (LLM)"
    ])

    # Tab 1: Stopwords & Advanced
    with tab_stopwords:
        st.subheader("Stopwords: Manage Built-In & Custom")
        new_word = st.text_input("Add a word to remove", key="new_word_tab1")
        if st.button("Add Word to Remove", key="add_btn_tab1"):
            if new_word.strip():
                st.session_state.custom_stopwords.add(new_word.strip().lower())

        if st.session_state.custom_stopwords:
            st.write("#### Currently Removed (Custom) Words")
            remove_list = sorted(st.session_state.custom_stopwords)
            for w in remove_list:
                col1, col2 = st.columns([4,1])
                col1.write(w)
                if col2.button(f"Remove '{w}'", key=f"rm_{w}_tab1"):
                    st.session_state.custom_stopwords.remove(w)
        else:
            st.info("No custom stopwords yet.")

        df_tab = df.copy()
        df_tab['finalText'] = df_tab['cleanedText'].apply(lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords))
        df_tab['sentiment'] = df_tab['finalText'].apply(analyze_sentiment)
        df_tab['polarity'] = df_tab['sentiment'].apply(lambda s: s.polarity)
        df_tab['subjectivity'] = df_tab['sentiment'].apply(lambda s: s.subjectivity)

        st.subheader("Articles Table")
        st.dataframe(df_tab[['title','publication','author','publishedAt','description','polarity','subjectivity']])

        st.subheader("Top Words (Frequency)")
        wordFreq = compute_word_frequency(df_tab['finalText'])
        freq_items = sorted(wordFreq.items(), key=lambda x: x[1], reverse=True)
        topN = 50
        top_words = freq_items[:topN]

        if top_words:
            words, counts = zip(*top_words)
            freq_df = pd.DataFrame({'word': words, 'count': counts})
            freq_df = freq_df.sort_values(by='count', ascending=False)
            st.bar_chart(freq_df.set_index('word'))
        else:
            st.write("No words left after removing all stopwords!")

        st.subheader("Word Frequency Table")
        freq_df_all = pd.DataFrame(freq_items, columns=["Word", "Count"])
        st.dataframe(freq_df_all)

        st.subheader("Word Cloud")
        all_text = ' '.join(df_tab['finalText'].tolist())
        if all_text.strip():
            fig = create_wordcloud(all_text, stopwords=set())
            st.pyplot(fig)
        else:
            st.write("No text available for word cloud after removing stopwords!")

        # Lemmas, Bigrams, Trigrams
        st.subheader("Advanced: Lemmas, Bigrams, Trigrams")
        df_advanced = df_tab.copy()
        df_advanced['lemmas'] = df_advanced['finalText'].apply(lemmatise_text_spacy)

        st.markdown("### Lemmas")
        lemma_freq = compute_word_frequency(df_advanced['lemmas'])
        lemma_items = sorted(lemma_freq.items(), key=lambda x: x[1], reverse=True)
        st.write("#### Lemma Frequency (Top 20)")
        st.write(lemma_items[:20])
        all_lemmas = " ".join(df_advanced['lemmas'])
        fig_lem = create_wordcloud(all_lemmas, stopwords=set())
        st.pyplot(fig_lem)

        st.markdown("### Bigrams")
        df_advanced['bigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=2))
        bigram_freq = compute_word_frequency(df_advanced['bigrams'])
        bigram_items = sorted(bigram_freq.items(), key=lambda x: x[1], reverse=True)
        st.write("#### Bigram Frequency (Top 20)")
        st.write(bigram_items[:20])
        all_bigrams = " ".join(df_advanced['bigrams'])
        fig_big = create_wordcloud(all_bigrams, stopwords=set())
        st.pyplot(fig_big)

        st.markdown("### Trigrams")
        df_advanced['trigrams'] = df_advanced['lemmas'].apply(lambda txt: generate_ngrams(txt, n=3))
        trigram_freq = compute_word_frequency(df_advanced['trigrams'])
        trigram_items = sorted(trigram_freq.items(), key=lambda x: x[1], reverse=True)
        st.write("#### Trigram Frequency (Top 20)")
        st.write(trigram_items[:20])
        all_trigrams = " ".join(df_advanced['trigrams'])
        fig_tri = create_wordcloud(all_trigrams, stopwords=set())
        st.pyplot(fig_tri)

    # Tab 2: NER
    with tab_ner:
        st.subheader("Named Entity Recognition (NER)")
        entity_counts = {}
        for idx, row in df.iterrows():
            raw_title = row.title or ''
            raw_desc = row.description or ''
            raw_cont = row.content or ''
            doc = nlp(f"{raw_title} {raw_desc} {raw_cont}")
            for ent in doc.ents:
                key = (ent.text, ent.label_)
                entity_counts[key] = entity_counts.get(key, 0) + 1

        sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        st.write(f"**Total unique entities found**: {len(sorted_entities)}")

        topN_ents = 30
        top_entities = sorted_entities[:topN_ents]
        rows = []
        for (text_label, count) in top_entities:
            (ent_text, ent_label) = text_label
            rows.append([ent_text, ent_label, count])
        df_ents = pd.DataFrame(rows, columns=["Entity", "Label", "Count"])
        st.write("### Top Entities (Table)")
        st.dataframe(df_ents)

        combined_keys = [f"{t[0]} ({t[1]})" for t, c in top_entities]
        combined_counts = [c for t, c in top_entities]
        st.write("### Top Entities (Bar Chart)")
        if combined_keys:
            chart_df = pd.DataFrame({"entity": combined_keys, "count": combined_counts})
            chart_df = chart_df.sort_values(by='count', ascending=False)
            st.bar_chart(chart_df.set_index('entity'))
        else:
            st.info("No entities found to display.")

        # Word Cloud of entity text
        all_ents_text = []
        for (ent_text, ent_label), count in top_entities:
            ent_underscored = ent_text.replace(" ", "_")
            all_ents_text.extend([ent_underscored] * count)

        if all_ents_text:
            st.write("### Word Cloud of Entity Text")
            entity_text_big_string = " ".join(all_ents_text)
            fig_ent_wc = create_wordcloud(entity_text_big_string, stopwords=set())
            st.pyplot(fig_ent_wc)
        else:
            st.info("No entity text available for word cloud.")

    # Tab 3: Gensim Topic Modelling
    with tab_topics:
        st.subheader("Topic Modelling (Gensim LDA) + Interactive Networks")
        df_tab_for_topics = df.copy()
        df_tab_for_topics['finalText'] = df_tab_for_topics['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        num_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=5, step=1)
        num_words = st.number_input("Number of Top Words per Topic", min_value=5, max_value=30, value=10, step=1)
        passes = st.slider("Number of Training Passes (LDA)", min_value=1, max_value=20, value=10, step=1)

        if st.button("Run Topic Modelling (Gensim)"):
            with st.spinner("Running LDA..."):
                try:
                    docs = df_tab_for_topics['finalText'].tolist()
                    lda_model, corpus, dictionary = run_lda_topic_model(
                        docs, 
                        num_topics=num_topics, 
                        passes=passes, 
                        num_words=num_words
                    )
                    st.success("Gensim LDA complete!")

                    st.write("### Discovered Topics:")
                    topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                    for i, topic in topics:
                        st.write(f"**Topic {i}**")
                        topic_terms = [term for term, _ in topic]
                        st.write(", ".join(topic_terms))
                        st.write("---")

                    st.write("### Interactive Topic Networks")
                    for i, topic in topics:
                        st.subheader(f"Topic {i} Network")
                        net = create_topic_pyvis_network(i, topic)
                        display_pyvis_network(net, i)
                except Exception as ex:
                    st.error(f"Error running Gensim LDA: {ex}")

    # Tab 4: Keyword Extraction (TF-IDF)
    with tab_keywords:
        st.subheader("Keyword Extraction (TF-IDF)")
        df_tab_for_keywords = df.copy()
        df_tab_for_keywords['finalText'] = df_tab_for_keywords['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        topN_keywords = st.number_input("Number of Keywords to Extract", min_value=5, max_value=50, value=10, step=1)

        if st.button("Run Keyword Extraction"):
            with st.spinner("Extracting keywords..."):
                try:
                    docs = df_tab_for_keywords['finalText'].tolist()
                    if not docs:
                        st.warning("No documents found.")
                    else:
                        top_keywords = extract_keywords_tfidf(docs, top_n=topN_keywords)
                        st.success("Keyword extraction complete!")
                        st.write("### Top Keywords (by TF-IDF)")
                        top_keywords_sorted = sorted(top_keywords, key=lambda x: x[1], reverse=True)
                        df_kw = pd.DataFrame(top_keywords_sorted, columns=["Keyword", "TF-IDF Score"])
                        st.dataframe(df_kw)

                        st.write("#### TF-IDF Bar Chart")
                        if not df_kw.empty:
                            chart_df = df_kw.set_index("Keyword")
                            st.bar_chart(chart_df)
                except Exception as ex:
                    st.error(f"Error extracting keywords: {ex}")

    # Tab 5: Clustering & Classification
    # with tab_clustering:
    #     st.subheader("Clustering & Classification (K-Means Demo)")
    #     df_tab_for_clustering = df.copy()
    #     df_tab_for_clustering['finalText'] = df_tab_for_clustering['cleanedText'].apply(
    #         lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
    #     )

    #     num_clusters = st.number_input("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
    #     show_top_cluster_terms = st.checkbox("Show top terms in each cluster?", value=True)

    #     if st.button("Run Clustering"):
    #         with st.spinner("Running K-Means clustering..."):
    #             try:
    #                 docs = df_tab_for_clustering['finalText'].tolist()
    #                 if not docs:
    #                     st.warning("No documents found to cluster.")
    #                 else:
    #                     labels, kmeans_model, vectorizer, X = cluster_documents_kmeans(docs, num_clusters=num_clusters)
    #                     st.success("K-Means Clustering complete!")

    #                     df_cluster = df_tab_for_clustering[['title', 'publication', 'finalText']].copy()
    #                     df_cluster['cluster'] = labels
    #                     st.write("### Documents & Their Assigned Clusters")
    #                     st.dataframe(df_cluster)

    #                     if show_top_cluster_terms:
    #                         st.write("### Top Terms by Cluster Centroid")
    #                         cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)
    #                         for cid, terms in cluster_top_terms.items():
    #                             st.markdown(f"**Cluster {cid}**")
    #                             top_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
    #                             st.write(top_str)
    #             except Exception as ex:
    #                 st.error(f"Error clustering: {ex}")




        with tab_clustering:
            st.subheader("Clustering & Classification (K-Means Demo)")
            df_tab_for_clustering = df.copy()
            df_tab_for_clustering['finalText'] = df_tab_for_clustering['cleanedText'].apply(
                lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
            )

            num_clusters = st.number_input("Number of Clusters (K-Means)", min_value=2, max_value=10, value=3, step=1)
            show_top_cluster_terms = st.checkbox("Show top terms in each cluster?", value=True)

            if st.button("Run Clustering"):
                with st.spinner("Running K-Means clustering..."):
                    try:
                        docs = df_tab_for_clustering['finalText'].tolist()
                        if not docs:
                            st.warning("No documents found to cluster.")
                        else:
                            # IMPORTANT: Correct unpacking order below
                            kmeans_model, vectorizer, X, labels = cluster_documents_kmeans(
                                docs, 
                                num_clusters=num_clusters
                            )
                            st.success("K-Means Clustering complete!")

                            df_cluster = df_tab_for_clustering[['title', 'publication', 'finalText']].copy()
                            df_cluster['cluster'] = labels
                            st.write("### Documents & Their Assigned Clusters")
                            st.dataframe(df_cluster)

                            if show_top_cluster_terms:
                                st.write("### Top Terms by Cluster Centroid")
                                cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)
                                for cid, terms in cluster_top_terms.items():
                                    st.markdown(f"**Cluster {cid}**")
                                    top_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
                                    st.write(top_str)
                    except Exception as ex:
                        st.error(f"Error clustering: {ex}")







        st.write("---")
        st.write("## Classification (Placeholder)")
        st.info("""Typically, you'd need labelled data to train a supervised model. 
                This is just a placeholder for a future extension.""")

    # Tab 6: Sentiment Visualisation
    def compute_color_for_polarity(p):
        if p == 0:
            return None
        p = max(-1, min(1, p))
        if p < 0:
            intensity = abs(p)
            r1, g1, b1 = 255, 201, 201
            r2, g2, b2 = 255,   0,   0
            r = int(r1 + (r2-r1)*intensity)
            g = int(g1 + (g2-g1)*intensity)
            b = int(b1 + (b2-b1)*intensity)
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            intensity = p
            r1, g1, b1 = 200, 247, 197
            r2, g2, b2 = 0,   176, 0
            r = int(r1 + (r2-r1)*intensity)
            g = int(g1 + (g2-g1)*intensity)
            b = int(b1 + (b2-b1)*intensity)
            return f"#{r:02x}{g:02x}{b:02x}"

    def compute_color_for_subjectivity(s):
        if s == 0:
            return None
        r1, g1, b1 = 213, 243, 254
        r2, g2, b2 = 0,   119, 190
        r = int(r1 + (r2-r1)*s)
        g = int(g1 + (g2-g1)*s)
        b = int(b1 + (b2-b1)*s)
        return f"#{r:02x}{g:02x}{b:02x}"

    def highlight_word_polarity(word):
        from textblob import TextBlob
        p = TextBlob(word).sentiment.polarity
        col = compute_color_for_polarity(p)
        if col is None:
            return f"<span style='margin:2px;'>{word}</span>"
        else:
            return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"

    def highlight_word_subjectivity(word):
        from textblob import TextBlob
        s = TextBlob(word).sentiment.subjectivity
        col = compute_color_for_subjectivity(s)
        if col is None:
            return f"<span style='margin:2px;'>{word}</span>"
        else:
            return f"<span style='background-color:{col}; margin:2px;'>{word}</span>"

    def highlight_text_polarity(full_text):
        words = full_text.split()
        highlighted = [highlight_word_polarity(w) for w in words]
        return " ".join(highlighted)

    def highlight_text_subjectivity(full_text):
        words = full_text.split()
        highlighted = [highlight_word_subjectivity(w) for w in words]
        return " ".join(highlighted)

    with tab_sentviz:
        st.subheader("Per-Article Sentiment Explanation")
        df_tab_sent = df.copy()
        df_tab_sent['finalText'] = df_tab_sent['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        df_tab_sent['sentiment'] = df_tab_sent['finalText'].apply(analyze_sentiment)
        df_tab_sent['polarity'] = df_tab_sent['sentiment'].apply(lambda s: s.polarity)
        df_tab_sent['subjectivity'] = df_tab_sent['sentiment'].apply(lambda s: s.subjectivity)

        if df_tab_sent.empty:
            st.warning("No articles to display. Please fetch some articles first.")
        else:
            article_indices = df_tab_sent.index.tolist()
            chosen_idx = st.selectbox("Choose article index:", article_indices)

            row = df_tab_sent.loc[chosen_idx]

            st.write("### Article Metadata")
            details = {
                "Title": row.get('title', 'N/A'),
                "Publication": row.get('publication', 'N/A'),
                "Published": row.get('publishedAt', 'N/A'),
                "Polarity": round(row.get('polarity', 0), 3),
                "Subjectivity": round(row.get('subjectivity', 0), 3)
            }
            meta_df = pd.DataFrame([details])
            st.table(meta_df)

            final_text = row['finalText'] or ""

            st.write("### Polarity Highlighter (Word-Level)")
            pol_html = highlight_text_polarity(final_text)
            st.markdown(pol_html, unsafe_allow_html=True)

            st.write("### Subjectivity Highlighter (Word-Level)")
            subj_html = highlight_text_subjectivity(final_text)
            st.markdown(subj_html, unsafe_allow_html=True)

    # Tab 7: Detailed Topics & Clusters (scikit-learn approach)
    with tab_topics_clusters:
        st.subheader("Detailed Topic Discovery & Clustering (scikit-learn)")

        st.write("""
        This tab performs scikit-learn LDA and K-Means on the same final text, 
        storing the results so the next tab can build a structured LLM prompt.
        """)

        df_for_analysis = df.copy()
        df_for_analysis['finalText'] = df_for_analysis['cleanedText'].apply(
            lambda x: apply_stopwords_union(x, st.session_state.custom_stopwords)
        )

        docs = df_for_analysis['finalText'].tolist()
        if not docs:
            st.warning("No documents available. Please fetch or clean your articles.")
        else:
            num_topics = st.number_input("Number of LDA Topics", 2, 20, 5, 1)
            top_words_lda = st.number_input("Top Words per Topic (LDA)", 5, 30, 10, 1)
            lda_max_iter = st.slider("LDA Max Iterations", 5, 100, 10, 5)

            num_clusters = st.number_input("Number of K-Means Clusters", 2, 20, 5, 1)
            top_words_kmeans = st.number_input("Top Words per Cluster (K-Means)", 5, 30, 10, 1)

            if st.button("Run Analysis (Store Results)"):
                with st.spinner("Performing LDA and K-Means..."):
                    # LDA
                    lda_model, doc_topic_matrix, lda_feature_names = run_sklearn_lda_topic_modelling(
                        docs, 
                        n_topics=num_topics, 
                        n_top_words=top_words_lda, 
                        max_iter=lda_max_iter
                    )
                    topic_top_words = display_top_words_for_lda(lda_model, lda_feature_names, n_top_words=top_words_lda)

                    # For each doc, find dominant topic
                    dominant_topics = doc_topic_matrix.argmax(axis=1)
                    df_for_analysis['LDA_Topic'] = dominant_topics

                    # K-Means
                    # kmeans_model, labels, kmeans_feature_names, X = run_kmeans_clustering_sklearn(
                    #     docs, 
                    #     n_clusters=num_clusters, 
                    #     n_top_words=top_words_kmeans
                    # )

                    kmeans_model, vectorizer, X, labels = cluster_documents_kmeans(
                        docs, 
                        num_clusters=3
                    )

                    df_for_analysis['KMeans_Cluster'] = labels

                    # cluster_top_terms = get_top_terms_per_cluster(kmeans_model, kmeans_feature_names, n_top_words=top_words_kmeans)
                    
                    cluster_top_terms = get_top_terms_per_cluster(kmeans_model, vectorizer, num_terms=10)

                    st.success("Analysis complete! Storing results for LLM usage...")

                    st.write("### LDA Topics & Top Words")
                    for topic_idx, words in topic_top_words.items():
                        st.markdown(f"**Topic {topic_idx}**: {', '.join(words)}")

                    st.write("### Document → Topic Mapping (LDA)")
                    st.dataframe(df_for_analysis[['title','publication','finalText','LDA_Topic']])

                    st.write("### K-Means Clusters")
                    st.write("#### Document → Cluster Mapping")
                    st.dataframe(df_for_analysis[['title','publication','finalText','KMeans_Cluster']])

                    st.write("#### Top Terms in Each K-Means Cluster")
                    for cid, term_list in cluster_top_terms.items():
                        top_words_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in term_list])
                        st.markdown(f"**Cluster {cid}**: {top_words_str}")

                    # Save to session_state
                    st.session_state.df_for_thematic = df_for_analysis
                    st.session_state.lda_topic_top_words = topic_top_words
                    st.session_state.cluster_top_terms = cluster_top_terms
                    st.session_state.topic_assignments = dominant_topics
                    st.session_state.cluster_assignments = labels


    # -------------------------------------------------------------
    # TAB 8: Narratives (LLM) – Thematic Summaries
    # -------------------------------------------------------------
    with tab_narratives:
        st.subheader("Narratives (LLM) – Thematic Summaries (Detailed)")

        st.write("""
        This tab sends the raw data **plus** the thematical analyses (topics, clusters, key words) 
        to ChatGPT, asking for a **more detailed** explanation of the key narratives.
        
        Additionally, you can **download** the entire LLM prompt in JSON format 
        if you want to inspect it or store it.
        """)

        if st.session_state.df_for_thematic.empty:
            st.warning("No thematical results found. Please run 'Detailed Topics & Clusters' first.")
        else:
            df_them = st.session_state.df_for_thematic.copy()
            # Check if LLM key is valid
            if not st.session_state.llm_key_validated or not st.session_state.llm_api_key:
                st.warning("Please provide and validate your ChatGPT API key in the sidebar.")
            else:
                # 1) LDA topic info
                topic_info_str = "## LDA Topics:\n"
                for t_id, words in st.session_state.lda_topic_top_words.items():
                    topic_info_str += f"Topic {t_id}: {', '.join(words)}\n"

                # 2) K-Means cluster info
                cluster_info_str = "## K-Means Clusters:\n"
                for c_id, words in st.session_state.cluster_top_terms.items():
                    cluster_info_str += f"Cluster {c_id} top words: "
                    cluster_info_str += ", ".join([f"{w[0]}({w[1]:.3f})" for w in words])
                    cluster_info_str += "\n"

                # 3) Document-level assignments (limit to 5 docs for brevity)
                doc_samples = df_them.head(5)
                doc_info_str = "## Document Assignments:\n"
                for i, row in doc_samples.iterrows():
                    doc_info_str += f"\n--- Document idx={i} ---\n"
                    doc_info_str += f"Title: {row.get('title','N/A')}\n"
                    doc_info_str += f"LDA_Topic: {row.get('LDA_Topic','N/A')}, "
                    doc_info_str += f"KMeans_Cluster: {row.get('KMeans_Cluster','N/A')}\n"
                    text_excerpt = row.get('finalText','')[:300]
                    doc_info_str += f"Text excerpt: {text_excerpt}...\n"

                # Show user the data
                st.write("### Data We'll Send to the LLM")
                st.text(topic_info_str + "\n\n" + cluster_info_str + "\n\n" + doc_info_str)

                # Build the final prompt
                system_msg = (
                    "You are a helpful data analyst. "
                    "We have results from a topic/cluster analysis. "
                    "Please read the info below and produce a cohesive narrative. "
                    "Focus on describing the key themes or storylines that emerge, and "
                    "provide a more thorough, in-depth analysis. Elaborate on how "
                    "these topics and clusters interrelate, discussing any nuanced "
                    "differences or patterns you see."
                )

                user_msg = (
                    f"THEMATIC ANALYSIS DATA:\n\n{topic_info_str}\n\n{cluster_info_str}\n\n{doc_info_str}\n\n"
                    "INSTRUCTION:\n"
                    "Please provide a **more detailed** analysis of the key narratives found in these topics/clusters. "
                    "Explain how the articles interrelate or differ, any major themes, perspectives, or storylines, "
                    "and any relevant patterns that stand out. "
                    "Feel free to be as comprehensive as possible."
                )

                # For the download button, we'll package these messages into JSON
                import json
                prompt_payload = {
                    "system": system_msg,
                    "user": user_msg
                }
                prompt_json_str = json.dumps(prompt_payload, indent=2)

                st.download_button(
                    label="Download LLM Prompt as JSON",
                    data=prompt_json_str,
                    file_name="llm_prompt.json",
                    mime="application/json"
                )

                st.write("""
                Press the button below to send the above data to ChatGPT, 
                requesting a more **detailed** thematic analysis.
                """)

                if st.button("Generate Detailed LLM Narrative"):
                    with st.spinner("Asking ChatGPT..."):
                        try:
                            openai.api_key = st.session_state.llm_api_key
                            # Old method (for openai<1.0.0):
                            response = openai.ChatCompletion.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "system", "content": system_msg},
                                    {"role": "user", "content": user_msg}
                                ],
                                max_tokens=1500,
                                temperature=0.7
                            )
                            result = response["choices"][0]["message"]["content"]

                            st.write("### LLM Narrative Summary (Detailed)")
                            st.write(result)

                        except Exception as ex:
                            st.error(f"LLM Error: {ex}")


if __name__ == "__main__":
    main()
























