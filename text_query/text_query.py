import re
import itertools as it
import spacy
import scispacy
from html import unescape
import copy

import scispacy
import spacy
from spacy.lemmatizer import Lemmatizer
# from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex

def custom_tokenizer(nlp):
    _quotes = CONCAT_QUOTES.replace("'", "")
    _infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[{al}])\.(?=[{au}])".format(al=ALPHA_LOWER, au=ALPHA_UPPER),
            r"(?<=[{a}])[,!?](?=[{a}])".format(a=ALPHA),
            r'(?<=[{a}])[:<>=](?=[{a}])'.format(a=ALPHA),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}])([{q}\)\]\(\[])(?=[{a}])".format(a=ALPHA, q=_quotes),
            r"(?<=[{a}])--(?=[{a}])".format(a=ALPHA),
            r"(?<=[{a}])-(?=[{a}])".format(a=ALPHA),  # add rule splitting words by hyphen
            r"(?<=[{a}])/(?=[{a}])".format(a=ALPHA),  # add rule splitting words by slash
            r"(?<=[0-9])-(?=[0-9])",
        ]
    )
    infix_re = compile_infix_regex(_infixes)

    return Tokenizer(nlp.vocab, prefix_search=nlp.tokenizer.prefix_search,
                                suffix_search=nlp.tokenizer.suffix_search,
                                infix_finditer=infix_re.finditer,
                                token_match=nlp.tokenizer.token_match,
                                rules=nlp.Defaults.tokenizer_exceptions)

def get_nlp():
    nlp = spacy.load('en_core_sci_md')
#     lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    lemmatizer = English.Defaults.create_lemmatizer()
    nlp.tokenizer = custom_tokenizer(nlp)
    
    return nlp, lemmatizer

def text_filter(s):
    try:
        s = re.sub('\\\\n',' ',s)
        s = re.sub('\\\\t',' ',s)
        s = re.sub("\\\\'","'",s)
        s = re.sub('^\s+','',s)
        s = re.sub('\s+$','',s)
        s = re.sub('\s+',' ',s)
        s = unescape(s)
        if len(s) == 0: s = None
    except: s = None
    
    return s

# uses spaCy
def split_tag_sentences(s, nlp, split_sentence=True, lemmatizer=None):
    doc = nlp(s)
    tagged_words = [{'word': str(w), 'tag': str(w.tag_), 'pos': str(w.pos_), 'dep': str(w.dep_), 'idx': int(w.idx)} for w in doc]
    if lemmatizer:
        for i,w in enumerate(tagged_words):
            tagged_words[i]['lemma'] = str(lemmatizer(w['word'].lower(), w['pos'])[0])
    
    if split_sentence:
        sents = []
        for sent in doc.sents:
            start_idx = int(tagged_words[sent.start]['idx'])
            try: end_idx = int(tagged_words[sent.end]['idx']-1)
            except: end_idx = len(s)

            for i in range(sent.start,sent.end):
                tagged_words[i]['idx'] -= start_idx

            sents.append({'sentence': tagged_words[sent.start:sent.end], 'string': s[start_idx:end_idx]})
    else:
        sents = [{'sentence': tagged_words, 'string': s}]
    
    return sents

def handle_sentence(s, nlp, split_sentence=True, lemmatizer=None, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    sentences = split_tag_sentences(s, nlp, lemmatizer=lemmatizer, split_sentence=split_sentence)
    result_sentences = []
    for i, sentence in enumerate(sentences):
        word_index = {}
        lemmas = []
        for j,w in enumerate(sentence['sentence']):
            word_index[j] = { 'word': w['word'], 
                              'type': "stop word" if w['word'] in stop_words else "word", 
                              'id': j, 
                              'dep': w['dep'],
                              'idx': ( w['idx'], w['idx']+len(w['word'])-1 ),
                              'pos': w['pos'],
                              'tag': w['tag']}
            if lemmatizer:
                if not w['lemma'].lower() == w['word'].lower(): 
                    lemmas.append({'parent_id': j, 'word': w['lemma']})
        
        ids = sorted(list(word_index.keys()))
        if len(ids)==0: continue
        
        conn = {None: {ids[0]}}
        j=0
        for j in range(1,len(ids)):
            try: conn[ids[j-1]].add(ids[j])
            except: conn[ids[j-1]] = {ids[j]}
        else: conn[ids[j]] = {None}

        for lemma in lemmas:
            parent_word = word_index[lemma['parent_id']]
            k = max(word_index.keys()) + 1
            word_index[k] = { 'word': lemma['word'], 
                                              'type': "stop word" if w['word'] in stop_words else "word", 
                                              'id': k, 
                                              'dep': parent_word['dep'],
                                              'pos': parent_word['pos'],
                                              'tag': parent_word['tag'],
                                              'parent': {lemma['parent_id']} }
            
            rev_conn = gen_rev_conn(conn)
            for j in rev_conn[lemma['parent_id']]:
                conn[j].add(k)
            
            rev_conn = gen_rev_conn(conn)
            conn[k] = conn[lemma['parent_id']].copy()
            
        result_sentences.append({'string': sentence['string'], 'words': word_index, 'conn': conn})
    return result_sentences

def rec_parse(node, inline_tags={'content', 'sup', 'sub', 'linkHtml', 'caption'}):
    tag = node.tag
    tag = re.sub('\{.*?\}','',tag)
    node_items = []
    
    s = text_filter(node.text)
    if not s is None: node_items.append(str(s))
    
    children = node[:] # shortcut for getchildren()
    for child in children:
        child_data = rec_parse(child, inline_tags=inline_tags)
        if child_data is None: continue 
        if child_data['tag'] in inline_tags: # if the child node is an inline node then remove the nesting
            for i in child_data['nodes']:
                node_items.append(i)
        else:
            node_items.append(child_data)
    
    s = text_filter(node.tail)
    if not s is None: node_items.append(str(s))
    
    if len(node_items) == 0: return None
    
    return {'tag': tag, 'nodes': node_items}

# join neighboring strings together, split them by sentence attaching the label "string" to each sentence. Split each sentence into words, tag the stop words. 
def rec_join_str(node, nlp, lemmatizer=None, split_sentence=True, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    if node is None: return None
    joined_items = []
    s = []
    for n in node['nodes']:
        if n is None: continue
        if isinstance(n,str): 
            s.append(str(n))
        else:
            if len(s) > 0:
                sentences = handle_sentence(" ".join(s), nlp, lemmatizer=lemmatizer, split_sentence=split_sentence, stop_words=stop_words)
                string_item = {'tag': "string", 'nodes': sentences}
                joined_items.append(copy.deepcopy(string_item))
                s = []
                
            joined_items.append((rec_join_str(n, nlp)))
    
    if len(s) > 0:
        sentences = handle_sentence(" ".join(s), nlp)
        string_item = {'tag': "string", 'nodes': sentences}
        joined_items.append(copy.deepcopy(string_item))
    
    if len(joined_items) == 0: return None
    
    return {'tag':node['tag'], 'nodes': joined_items}


def generate_kmers(l,n):
    for i in range(len(l)-n+1):
        yield tuple(sorted(l[i:i+n]))

def generate_ordered_kmers(l,n):
    for i in range(len(l)-n+1):
        yield l[i:i+n]

def gen_rev_conn(conn):
    rev_conn = {}
    for k,vs in conn.items():
        for v in vs:
            try: rev_conn[v].add(k)
            except: rev_conn[v] = {k}
    return rev_conn

def next_conn_skip_stop_words(sentence, start_id, stop_words):
    next_ids = set()
    if not start_id in sentence['conn']: return set()
    for next_id in sentence['conn'][start_id]:
        if next_id is None: 
            next_ids.add(next_id)
            continue
        if sentence['words'][next_id]['word'].lower() in stop_words:
            next_ids.update(next_conn_skip_stop_words(sentence, next_id, stop_words))
        else: next_ids.add(next_id)
    return next_ids

def next_rev_conn_skip_stop_words(sentence, rev_conn, start_id, stop_words):
    next_ids = set()
    if not start_id in rev_conn: return set()
    for next_id in rev_conn[start_id]:
        if next_id is None: 
            next_ids.add(next_id)
            continue
        if sentence['words'][next_id]['word'].lower() in stop_words:
            next_ids.update(next_rev_conn_skip_stop_words(sentence, rev_conn, next_id, stop_words))
        else: next_ids.add(next_id)
    return next_ids

def rec_conn_gen_kmers(sentence, kmer, n, stop_words):
    conn = sentence['conn']
    if len(kmer) >= n: return {tuple(kmer)}
    if not kmer[-1] in conn.keys(): return set()
    results = set()
    for conn_id in next_conn_skip_stop_words(sentence, kmer[-1], stop_words):
        if conn_id is None: continue
        results.update(rec_conn_gen_kmers(sentence, kmer + [conn_id], n, stop_words))
    return results

def conn_gen_kmers(sentence, n, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    kmers = set()
    for word_id in sentence['words'].keys():
        if sentence['words'][word_id]['word'].lower() in stop_words: continue
        kmers.update(rec_conn_gen_kmers(sentence, [word_id], n, stop_words))
    return kmers

def rec_gen_sentences(conn, sentence=[None]):
    #if not sentence[-1] in conn: return {tuple(sentence)}
    sentences = set()
    for next_id in conn[sentence[-1]]:
        if next_id is None: 
            sentences.add(tuple(sentence[1:]))
            continue
        sentences.update(rec_gen_sentences(conn, sentence=sentence+[next_id]))
    return sentences


def expand_brackets(sentence):
    conn = sentence['conn']
    rev_conn = {}
    for k,vs in conn.items():
        for v in vs:
            try: rev_conn[v].add(k)
            except: rev_conn[v] = {k}
    
    # find brackets     
    words = sentence['words']
    open_brackets = {k for k,v in words.items() if v['word'] == '('}
    close_brackets = {k for k,v in words.items() if v['word'] == ')'}
    used_brackets = set()
    brackets = set()
    for open_bracket in sorted(open_brackets, reverse=True):
        try: close_bracket = sorted({i for i in (close_brackets - used_brackets) if i > open_bracket})[0]
        except: continue
        brackets.add((open_bracket, close_bracket))
        used_brackets.add(close_bracket)
    
    # add conns based on brackets
    # skip brackets
    for bracket_id in (open_brackets | close_brackets):
        # try: 
        #     previous_ids = rev_conn[bracket_id]
        #     next_ids = conn[bracket_id]
        # except: continue
        previous_ids = rev_conn[bracket_id]
        next_ids = conn[bracket_id]
        for id1,id2 in it.product(previous_ids, next_ids):
            conn[id1].add(id2)
            
    # skip bracket contents
    for open_bracket, close_bracket in brackets:
        # try: 
        #     previous_ids = rev_conn[open_bracket]
        #     next_ids = conn[close_bracket]
        # except: continue
        previous_ids = rev_conn[open_bracket]
        next_ids = conn[close_bracket]
        for id1,id2 in it.product(previous_ids, next_ids):
            conn[id1].add(id2)
    
    sentence['conn'] = conn
    
    return sentence

def expand_hyphen_slash(sentence):
    conn = sentence['conn']
    rev_conn = gen_rev_conn(conn)
    
    # find all hyphenated/slash words
    for w_i,word in sentence['words'].items():
        if word['word'] in {'-', '/'}:
            # get previous and next words
            try: prev_words = rev_conn[w_i]
            except: prev_words = set()
            try: next_words = conn[w_i]
            except: next_words = set()
            
            # go one more step forward and back
            prev_prev_words = set()
            for w in prev_words:
                try: prev_prev_words.update(rev_conn[w])
                except: continue
            next_next_words = set()
            for w in next_words:
                try: next_next_words.update(conn[w])
                except: continue
            
            #  prev --> next
            for p in prev_words:
                for n in next_words:
                    conn[p].add(n)
            #  prev_prev --> next      
            for pp in prev_prev_words:
                for n in next_words:
                    conn[pp].add(n)
            #  prev --> next_next
            for p in prev_words:
                for nn in next_next_words:
                    conn[p].add(nn)
                
            rev_conn = gen_rev_conn(conn)  # update rev_conn
            
    sentence['conn'] = conn
    return sentence

# def expand_slash(sentence):
#     conn = sentence['conn']
#     rev_conn = gen_rev_conn(conn)
    
#     # find all slash words
#     words = list(sentence['words'].values())
#     new_words = []
#     for word in words:
#         if '/' in set(word['word']): # check if slashes are present (old: regex matching '^[^/]+/([^/]+/)*[^/]+$')
#             split_words = word['word'].split('/')
#             split_words = [w for w in split_words if len(w)>0]
#             if len(split_words) <= 1: continue
                
#             max_id = max({w['id'] for w in (words + new_words)}) # for generating new IDs
            
#             for i,w in enumerate(split_words):
#                 new_words.append({'id': max_id+i+1, 'word': w, 'type': word['type'], 'tag': word['tag'], 'parent': {word['id']}}) # add the new word
#                 # generate the new connections to that word (copy them from the original word)
#                 # try: conn[max_id+i+1] = set(conn[word['id']])
#                 # except: pass
#                 conn[max_id+i+1] = set(conn[word['id']])
#                 if word['id'] in rev_conn:
#                     for word_id in rev_conn[word['id']]:
#                         conn[word_id].add(max_id+i+1)
#                 rev_conn = gen_rev_conn(conn)
    
#     sentence['words'] = {w['id']: w for w in words+new_words}
#     sentence['conn'] = conn
#     return sentence

# def expand_hyphen(sentence):
#     conn = sentence['conn']
#     rev_conn = gen_rev_conn(conn)
    
#     # find all hyphenated words
#     words = list(sentence['words'].values())
#     new_words = []
#     for word in words:
#         if '-' in set(word['word']): # check if hyphens are present
#             split_words = word['word'].split('-')
#             split_words = [w for w in split_words if len(w)>0]
#             if len(split_words) <= 1: continue
                
#             max_id = max({w['id'] for w in (words + new_words)}) # for generating new IDs
            
#             # add new words
#             new_words.append({'id': max_id+1, 'word': split_words[0], 'type': word['type'], 'tag': word['tag'], 'parent': {word['id']}})
#             new_words.append({'id': max_id+len(split_words), 'word': split_words[-1], 'type': word['type'], 'tag': word['tag'], 'parent': {word['id']}})
#             for i,w in enumerate(split_words[1:-1]):
#                 new_words.append({'id': max_id+i+2, 'word': w, 'type': word['type'], 'tag': word['tag'], 'parent': {word['id']}}) # add the new word
            
#             # add begining connections
#             if word['id'] in rev_conn.keys():
#                 for word_id in rev_conn[word['id']]:
#                     conn[word_id].add(max_id+1)
            
#             # try: conn[max_id+len(split_words)] = set(conn[word['id']]) # add end connections
#             # except: pass
#             conn[max_id+len(split_words)] = set(conn[word['id']]) # add end connections
            
#             # add inbetween connections
#             for i in range(1,len(split_words)):
#                 try: conn[max_id+i].add(max_id+i+1)
#                 except: conn[max_id+i] = {max_id+i+1}
                
#             rev_conn = gen_rev_conn(conn)
            
#     sentence['words'] = {w['id']: w for w in words+new_words}
#     sentence['conn'] = conn
#     return sentence

def expand_lists(sentence):
    conn = sentence['conn']
    sentences_ids = rec_gen_sentences(conn)

    # detect lists in each of these sentences, and make the nessesary adjustments
    for sentence_ids in sentences_ids:
        tags = [f"{i}{sentence['words'][i]['tag']}" for i in sentence_ids]
        s_pos_map = {word_id:i for i,word_id in enumerate(sentence_ids)}
        text_lists = re.findall('((?:(?:\d+(?:JJ|DT|NN|NNS|\?))+\d+(?:,|CC))+(?:\d+(?:JJ|DT|NN|NNS|\?))+)', "".join(tags)) # find all the sentences
        # generate all the versions of each list
        all_combs = []
        for text_list in text_lists:
            head_and_tails_ids = set()
                
            heads = re.findall('^((?:\d+(?:JJ|DT|NN|NNS|CC|\?))+)(?:\d+,|$)', text_list)
            for i,h in enumerate(heads):
                hs = re.findall('(?:(\d+)(?:JJ|DT|NN|NNS|\?))', h)
                heads[i] = [int(a) for a in hs]
                head_and_tails_ids.update(heads[i])

            tails = re.findall('(?:\d+,|^)((?:\d+(?:JJ|DT|NN|NNS|CC|\?))+)$', text_list)
            try: 
                tails = tails[0]
                tails = re.findall('((?:\d+(?:JJ|DT|NN|NNS|\?))+)', tails)
                for i,t in enumerate(tails):
                    ts = re.findall('(?:(\d+)(?:JJ|DT|NN|NNS|\?))', t)
                    tails[i] = [int(a) for a in ts]
                    head_and_tails_ids.update(tails[i])
            except: tails = []
            
            
            middle = re.findall('((?:\d+(?:JJ|DT|NN|NNS|\?))+)', text_list)
            middles = []
            for i,m in enumerate(middle):
                ms = re.findall('(?:(\d+)(?:JJ|DT|NN|NNS|\?))', m)
                ms = [int(a) for a in ms]
                ms = [a for a in ms if not a in head_and_tails_ids]
                if len(ms)>0: middles.append(ms)
                
            # make new connections
            for m in middles:
                m = [int(i) for i in m]
                # add begining to all heads
                for h in heads:
                    for h_id in h:
                        if s_pos_map[h_id] >= s_pos_map[m[0]]: continue
                        conn[h_id].add(m[0])
                # add end to all tails
                for t in tails:
                    for t_id in t:
                        if s_pos_map[m[-1]] >= s_pos_map[t_id]: continue
                        conn[m[-1]].add(t_id)
            
            for h in heads:
                for h_id in h:
                    for t in tails:
                        for t_id in t:
                            if s_pos_map[h_id] >= s_pos_map[t_id]: continue
                            try: conn[h_id].add(t_id)
                            except: print(h_id, conn)
                    
                
    sentence['conn'] = conn
    return sentence

def expand_sentence(original_sentence):
    # add other conns based on parentheses, hyphens, slashes and lists
    sentence = expand_lists(original_sentence)
    sentence = expand_brackets(sentence)
#     sentence = expand_slash(sentence)
#     sentence = expand_hyphen(sentence)
    sentence = expand_hyphen_slash(sentence)

    return sentence


def query_index_db(kmer, db_conn):
    kmer_str = re.sub('\'', '\'\'', repr(kmer)) # handle quotes
    results = db_conn.execute(f"select st.onto_id, st.predicate_type, st.source, st.string, st.expanded_sentences from (select * from (select * from kmers where kmer='{kmer_str}') km left join kmer_to_string on km.id=kmer_to_string.kmer_id) ks inner join (select * from strings) st on ks.string_id=st.id")
    for onto_id, predicate_type, source, string, sentence in results:
        yield {'source': source, 'onto_id': onto_id, 'predicate_type': predicate_type, 'string': string, 'sentence': eval(sentence)}

def kmer_query(sentence, query_f, f_args, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    # generate kmers from conn
    conn = sentence['conn']
    word_index = sentence['words']
    kmers = {}
    for n in range(1,4):
        kmers[n] = set()
        kmer_ids = conn_gen_kmers(sentence, n, stop_words=stop_words)
        for kmer_id in kmer_ids:
            kmer = tuple(sorted([word_index[k]['word'].lower() for k in kmer_id]))
            kmers[n].add(kmer)

    # get matches
    for n in range(1,4):
        for kmer in kmers[n]:
            # query the database here with the kmer
            matches = query_f(kmer, *f_args)
            for m in matches: yield m

# don't use!
def all_word_query(sentence, matches, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    sentence_words = {v['word'].lower() for k,v in sentence['words'].items()}
    for match in matches:
        for match_word_ids in rec_gen_sentences(match['sentence']['conn']):
            match_words = {match['sentence']['words'][i]['word'].lower() for i in match_word_ids}
            match_words = match_words - stop_words
            if len(match_words - sentence_words) == 0: 
                yield match
                break

def loc_query(sentence, matches, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    sentence['rev_conn'] = gen_rev_conn(sentence['conn'])
    for match in matches:
        match['sentence']['rev_conn'] = gen_rev_conn(match['sentence']['conn'])

        # get all matching paths
        # find all start points for match paths
        match_start_ids = next_conn_skip_stop_words(match['sentence'], None, stop_words)
        match_start_words = {}
        for word_id in match_start_ids:
            if word_id is None: continue
            word = match['sentence']['words'][word_id]['word'].lower()
            try: match_start_words[word].add(word_id)
            except: match_start_words[word] = {word_id}

        sentence_start_words = {}
        for match_start_word in match_start_words.keys():
            sentence_start_words[match_start_word] = {k for k,v in sentence['words'].items() if v['word'].lower() == match_start_word}

        start_ids = set()
        for word in match_start_words.keys():
            start_ids.update(it.product(sentence_start_words[word], match_start_words[word]))
        
        # for each of the start points, get the paths
        paths = set()
        for sentence_start_id, match_start_id in start_ids:
            #r = rec_conn_get_common_paths(sentence, match['sentence'], [(sentence_start_id,0)], [(match_start_id,0)], stop_words)
            r = conn_get_common_paths(sentence, match['sentence'], sentence_start_id, match_start_id, stop_words)
            #paths.update({p[0] for p in r})
            paths.update({tuple(p['sentence_path']) for p in r})

        good_paths = set()
        for path in paths:
            path_ids = tuple([p[0] for p in path])

            # filter out paths that have too many gaps
            gap_sum = sum([p[1] for p in path])
            if gap_sum > len(path_ids)/3: continue

            # filter out paths that don't make it to the end of the match
            #if len(path_ids & match['sentence']['rev_conn'][None]) == 0: continue
            
            good_paths.add(path_ids)

        if len(good_paths)>0: 
            match['paths'] = good_paths.copy()
            yield match

def rec_conn_get_common_paths(sentence, match_sentence, sentence_path, match_path, stop_words):
    paths = set()

    sentence_next_ids = next_conn_skip_stop_words(sentence, sentence_path[-1][0], stop_words)
    match_next_ids = next_conn_skip_stop_words(match_sentence, match_path[-1][0], stop_words)

    # allow for gaps
    sentence_next_next_ids = set()
    for sentence_next_id in sentence_next_ids:
        if sentence_next_id is None: continue
        sentence_next_next_ids.update(next_conn_skip_stop_words(sentence, sentence_next_id, stop_words))

    # form dictionary where each word has locs and the number of gaps needed to reach that loc
    sentence_next_words = {}
    for word_id in sentence_next_ids:
        if word_id is None: continue
        word = sentence['words'][word_id]['word'].lower()
        try: sentence_next_words[word].add((word_id,0))
        except: sentence_next_words[word] = {(word_id,0)}

    for word_id in sentence_next_next_ids:
        if word_id is None: continue
        word = sentence['words'][word_id]['word'].lower()
        try: sentence_next_words[word].add((word_id,1))
        except: sentence_next_words[word] = {(word_id,1)}

    match_next_words = {}
    for word_id in match_next_ids:
        if word_id is None:
            paths.add((tuple(sentence_path), tuple(match_path)))
            continue
        word = match_sentence['words'][word_id]['word'].lower()
        try: match_next_words[word].add((word_id,0))
        except: match_next_words[word] = {(word_id,0)}

    # generate list of words that both sentence and match can move to, and generate all the word_ids that fascilitate this transition to the next words
    next_words = set(match_next_words.keys()) & set(sentence_next_words.keys())
    next_ids = set()
    for word in next_words:
        next_ids.update(it.product(sentence_next_words[word], match_next_words[word]))

    for sentence_next_id, match_next_id in next_ids:
        paths.update(rec_conn_get_common_paths(sentence, match_sentence, sentence_path+[sentence_next_id], match_path+[match_next_id], stop_words))

    return paths

def conn_get_common_paths(sentence, match_sentence, sentence_start_id, match_start_id, stop_words):
    matched_paths = [{'sentence_path': [(sentence_start_id,0)], 'match_path': [(match_start_id,0)]},]
    confirmed_matched_paths = []
    while True:
        new_matched_paths = []
        for matched_path in matched_paths:
            sentence_next_ids = next_conn_skip_stop_words(sentence, matched_path['sentence_path'][-1][0], stop_words)
            match_next_ids = next_conn_skip_stop_words(match_sentence, matched_path['match_path'][-1][0], stop_words)

            # allow for gaps
            sentence_next_next_ids = set()
            for sentence_next_id in sentence_next_ids:
                if sentence_next_id is None: continue
                sentence_next_next_ids.update(next_conn_skip_stop_words(sentence, sentence_next_id, stop_words))

            # form dictionary where each word has locs and the number of gaps needed to reach that loc
            sentence_next_words = {}
            for word_id in sentence_next_ids:
                if word_id is None: continue
                word = sentence['words'][word_id]['word'].lower()
                try: sentence_next_words[word].add((word_id,0))
                except: sentence_next_words[word] = {(word_id,0)}

            for word_id in sentence_next_next_ids:
                if word_id is None: continue
                word = sentence['words'][word_id]['word'].lower()
                try: sentence_next_words[word].add((word_id,1))
                except: sentence_next_words[word] = {(word_id,1)}

            match_next_words = {}
            for word_id in match_next_ids:
                if word_id is None:
                    confirmed_matched_paths.append(matched_path)
                    continue
                word = match_sentence['words'][word_id]['word'].lower()
                try: match_next_words[word].add((word_id,0))
                except: match_next_words[word] = {(word_id,0)}

            # generate list of words that both sentence and match can move to, and generate all the word_ids that fascilitate this transition to the next words
            next_words = set(match_next_words.keys()) & set(sentence_next_words.keys())
            next_ids = set()
            for word in next_words:
                next_ids.update(it.product(sentence_next_words[word], match_next_words[word]))

            for sentence_next_id, match_next_id in next_ids:
                new_matched_paths.append({'sentence_path': matched_path['sentence_path']+[sentence_next_id], 
                                          'match_path': matched_path['match_path']+[match_next_id]})
        
        if len(new_matched_paths)>0: matched_paths = new_matched_paths
        else: break
        
    return confirmed_matched_paths

def query_sentence(sentence, db_conn):
    # do querying
    matches = kmer_query(sentence, query_index_db, [db_conn])            
    #matches = all_word_query(sentence, matches)
    matches = loc_query(sentence, matches)

    return list(matches)

def rec_query(node, loc, db_conn):
    results = {}
    if node['tag'] == "string":
        for i,sentence in enumerate(node['nodes']):
            new_loc = loc+[i]
            expanded_sentence = expand_sentence(sentence)
            matches = query_sentence(expanded_sentence, db_conn)

            if len(matches)>0:
                results[tuple(new_loc)] = {'loc': tuple(new_loc), 'sentence': sentence, 'matches': matches}
            
    else:
        for i,n in enumerate(node['nodes']):
            new_loc = loc+[i]
            rec_results = rec_query(n, new_loc, db_conn)
            for k,v in rec_results.items():
                results[k] = v
    
    return results

def onto_index_db_query(sections, db_conn):
    all_results = []
    for i, section in enumerate(sections):
        if not section is None: 
            results = rec_query(section, [i], db_conn)
            all_results.append(results)
    return all_results

def query_thesauruses(kmer, db_conn, sources):
    kmer_str = re.sub('\'', '\'\'', repr(kmer)) # handle quotes
    results = db_conn.execute(f"select st.onto_id, st.source, st.string, st.expanded_sentences from (select * from (select * from kmers where kmer='{kmer_str}') km left join kmer_to_string on km.id=kmer_to_string.kmer_id) ks inner join (select * from strings) st on ks.string_id=st.id")
    for onto_id, source, string, sentence in results:
        if source in sources: yield {'source': source, 'onto_id': onto_id, 'string': string, 'sentence': eval(sentence)}

def query_names_indexes(onto_id, db_conn):
    results = db_conn.execute(f"select source, string, predicate_type, sentences, expanded_sentences from strings where onto_id='{onto_id}'")
    for source, string, predicate_type, sentences, expanded_sentences in results:
        yield {'string': string, 'sentence': eval(expanded_sentences), 'original_sentence': eval(sentences)}

# def expand_thesaurus(sentence, matches, query_f, f_args, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
#     match_names = set()
#     for match in matches:
#         # for the added word parents 
#         match_path_ids = set()
#         match_start_ids = set()
#         match_end_ids = set()
#         for p in match['paths']:
#             match_path_ids.update(p)
#             match_start_ids.add(p[0])
#             match_end_ids.add(p[-1])

#         match_start_ids = match_start_ids - {None}
#         match_end_ids = match_end_ids - {None}

#         match_path_words = {sentence['words'][i]['word'].lower() for i in match_path_ids if not sentence['words'][i]['word'].lower() in stop_words}
        
#         names = query_f(match['onto_id'], *f_args)
        
#         # remove duplicates by standardising representation in hashed form
#         for name in names:
#             # check if the name adds any new words
#             name_words = {w['word'].lower() for k,w in name['sentence']['words'].items() if not w['word'].lower() in stop_words}
#             if len(name_words - match_path_words) > 0:
#                 name_sentence = {'words': {k:{'word': v['word'].lower(), 'id': v['id'], 'tag': v['tag'], 'type': v['type']} for k,v in name['sentence']['words'].items()}, 'conn': name['sentence']['conn']}
#                 match_names.add(repr({'sentence': name_sentence, 'match_path_ids': match_path_ids, 'match_start_ids': match_start_ids, 'match_end_ids': match_end_ids}))
        
#     for name in match_names:
#         name = eval(name)

#         max_id = max(sentence['words'].keys())  # needs to be updated each time
#         sentence['rev_conn'] = gen_rev_conn(sentence['conn'])  # needs to be updated each time
        
#         name_to_sentence_map = {}
        
#         # add words
#         for k,v in name['sentence']['words'].items():
#             i = v['id']
#             new_id = max_id+i+1
#             new_word = v.copy()
#             new_word['id'] = new_id
#             new_word['parent'] = name['match_path_ids']

#             name_to_sentence_map[i] = new_id
#             sentence['words'][new_id] = new_word

#         # add conn from name
#         name_rev_conn = gen_rev_conn(name['sentence']['conn'])
#         name_start_ids = name['sentence']['conn'][None] - {None}
#         name_end_ids = name_rev_conn[None] - {None}
        
#         for k,vs in name['sentence']['conn'].items():
#             if k is None:  # handle beginning of match
#                 for match_start_id in name['match_start_ids']:
#                     for prev_match_start_id in sentence['rev_conn'][match_start_id]:
#                         try: sentence['conn'][prev_match_start_id].update({name_to_sentence_map[v] for v in name_start_ids})
#                         except: sentence['conn'][prev_match_start_id] = {name_to_sentence_map[v] for v in name_start_ids}
#             else:
#                 new_k = name_to_sentence_map[k]
#                 for v in vs:
#                     if v is None:  # handle end of match
#                         for name_end_id in name_end_ids:
#                             name_end_id = name_to_sentence_map[name_end_id]
#                             for match_end_id in name['match_end_ids']:
#                                 try: sentence['conn'][name_end_id].update(sentence['conn'][match_end_id])
#                                 except: sentence['conn'][name_end_id] = set(sentence['conn'][match_end_id])
#                     else:  # handle the other words
#                         new_v = name_to_sentence_map[v]
#                         try: sentence['conn'][new_k].add(new_v)
#                         except: sentence['conn'][new_k] = {new_v}
                        
#     return sentence

def expand_thesaurus(sentence, matches, query_f, f_args, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    # organise thesaurus matches by location, just store the match code (not the string/sentence)
    match_locs = {}
    for match in matches:
        for p in match['paths']:
            try: match_locs[p].add(f"{match['source']};{match['onto_id']}")
            except: match_locs[p] = {f"{match['source']};{match['onto_id']}"}
    
    new_id = max([w['id'] for k,w in sentence['words'].items()]) + 1
    for path_ids, matches in match_locs.items():
        # add codes to sentence
        new_ids = set()
        for code in matches:
            sentence['words'][new_id] = {'id': new_id, 'word': code, 'parent': path_ids}
            new_ids.add(new_id)
            new_id+=1
        
        # add connections to the codes
        for code_id in new_ids:
            sentence['rev_conn'] = gen_rev_conn(sentence['conn'])
            for s_id in sentence['rev_conn'][path_ids[0]]:
                sentence['conn'][s_id].add(code_id)
            for s_id in sentence['conn'][path_ids[-1]]:
                try: sentence['conn'][code_id].add(s_id)
                except: sentence['conn'][code_id] = {s_id}
        sentence['rev_conn'] = gen_rev_conn(sentence['conn'])
        
    return sentence

def expand_index(sentence, kmer_query_f, names_query_f, kmer_query_args, names_query_args):
    # find matches from the thesaurus indexes
    matches = kmer_query(sentence, kmer_query_f, kmer_query_args, stop_words=set())
    matches = all_word_query(sentence, matches, stop_words=set())
    matches = loc_query(sentence, matches, stop_words=set())
    
    # substitute names from the matches into the sentence
    expanded_sentence = expand_thesaurus(sentence, matches, names_query_f, names_query_args)
    
    return expanded_sentence
