import re
import itertools as it
import spacy
import scispacy
from html import unescape
import copy

def text_filter(s):
    try:
        s = re.sub('\\\\n',' ',s)
        s = re.sub('\\\\t',' ',s)
        s = re.sub('^\s+','',s)
        s = re.sub('\s+$','',s)
        s = re.sub('\s+',' ',s)
        s = unescape(s)
        if len(s) == 0: s = None
    except: s = None
    
    return s

# uses spaCy
def split_tag_sentences(s, nlp):
    doc = nlp(s)
    tagged_words = [(str(w), str(w.tag_)) for w in doc]
    tagged_sents = []
    for sent in doc.sents:
        tagged_sents.append(tagged_words[sent.start:sent.end])
    return tagged_sents

def rec_parse(node):
    tag = node.tag
    tag = re.sub('\{.*?\}','',tag)
    node_items = []
    
    s = text_filter(node.text)
    if not s is None: node_items.append(str(s))
    
    children = node[:] # shortcut for getchildren()
    for child in children:
        child_data = rec_parse(child)
        if child_data is None: continue 
        if child_data['tag'] in {'content', 'sup', 'sub', 'linkHtml'}: # if the child node is an inline node then remove the nesting
            for i in child_data['nodes']:
                node_items.append(i)
        else:
            node_items.append(child_data)
    
    s = text_filter(node.tail)
    if not s is None: node_items.append(str(s))
    
    if len(node_items) == 0: return None
    
    return {'tag': tag, 'nodes': node_items}

def handle_sentence(s, nlp, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    sentences = split_tag_sentences(" ".join(s), nlp)
    for i, words in enumerate(sentences):
        word_index = {j: {'word': w[0], 
                          'type': "stop word" if w in stop_words else "word", 
                          'id': j, 
                          'tag': w[1]} for j, w in enumerate(words)}

        ids = sorted(list(word_index.keys()))
        conn = {None: {ids[0]}}
        j=0
        for j in range(1,len(ids)):
            try: conn[ids[j-1]].add(ids[j])
            except: conn[ids[j-1]] = {ids[j]}
        else: conn[ids[j]] = {None}

        sentences[i] = {'string': " ".join(s), 'words': word_index, 'conn': conn}
    return sentences

# join neighboring strings together, split them by sentence attaching the label "string" to each sentence. Split each sentence into words, tag the stop words. 
def rec_join_str(node, nlp):
    if node is None: return None
    joined_items = []
    s = []
    for n in node['nodes']:
        if n is None: continue
        if isinstance(n,str): 
            s.append(str(n))
        else:
            if len(s) > 0:
                sentences = handle_sentence(s, nlp)
                string_item = {'tag': "string", 'nodes': sentences}
                joined_items.append(copy.deepcopy(string_item))
                s = []
                
            joined_items.append((rec_join_str(n, nlp)))
    
    if len(s) > 0:
        sentences = handle_sentence(s, nlp)
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

def rec_gen_sentences(conn, sentence):
    #if not sentence[-1] in conn: return {tuple(sentence)}
    sentences = set()
    for next_id in conn[sentence[-1]]:
        if next_id is None: 
            sentences.add(tuple(sentence))
            continue
        sentences.update(rec_gen_sentences(conn, sentence+[next_id]))
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

def expand_slash(sentence):
    conn = sentence['conn']
    rev_conn = gen_rev_conn(conn)
    
    # find all slash words
    words = list(sentence['words'].values())
    new_words = []
    for word in words:
        if '/' in set(word['word']): # check if slashes are present (old: regex matching '^[^/]+/([^/]+/)*[^/]+$')
            split_words = word['word'].split('/')
            split_words = [w for w in split_words if len(w)>0]
            if len(split_words) <= 1: continue
                
            max_id = max({w['id'] for w in (words + new_words)}) # for generating new IDs
            
            for i,w in enumerate(split_words):
                new_words.append({'id': max_id+i+1, 'word': w, 'type': word['type'], 'tag': word['tag'], 'parent': {word['id']}}) # add the new word
                # generate the new connections to that word (copy them from the original word)
                # try: conn[max_id+i+1] = set(conn[word['id']])
                # except: pass
                conn[max_id+i+1] = set(conn[word['id']])
                if word['id'] in rev_conn:
                    for word_id in rev_conn[word['id']]:
                        conn[word_id].add(max_id+i+1)
                rev_conn = gen_rev_conn(conn)
    
    sentence['words'] = {w['id']: w for w in words+new_words}
    sentence['conn'] = conn
    return sentence

def expand_hyphen(sentence):
    conn = sentence['conn']
    rev_conn = gen_rev_conn(conn)
    
    # find all hyphenated words
    words = list(sentence['words'].values())
    new_words = []
    for word in words:
        if '-' in set(word['word']): # check if hyphens are present
            split_words = word['word'].split('-')
            split_words = [w for w in split_words if len(w)>0]
            if len(split_words) <= 1: continue
                
            max_id = max({w['id'] for w in (words + new_words)}) # for generating new IDs
            
            # add new words
            new_words.append({'id': max_id+1, 'word': split_words[0], 'type': word['type'], 'tag': word['tag'], 'parent': {word['id']}})
            new_words.append({'id': max_id+len(split_words), 'word': split_words[-1], 'type': word['type'], 'tag': word['tag'], 'parent': {word['id']}})
            for i,w in enumerate(split_words[1:-1]):
                new_words.append({'id': max_id+i+2, 'word': w, 'type': word['type'], 'tag': word['tag'], 'parent': {word['id']}}) # add the new word
            
            # add begining connections
            if word['id'] in rev_conn.keys():
                for word_id in rev_conn[word['id']]:
                    conn[word_id].add(max_id+1)
            
            # try: conn[max_id+len(split_words)] = set(conn[word['id']]) # add end connections
            # except: pass
            conn[max_id+len(split_words)] = set(conn[word['id']]) # add end connections
            
            # add inbetween connections
            for i in range(1,len(split_words)):
                try: conn[max_id+i].add(max_id+i+1)
                except: conn[max_id+i] = {max_id+i+1}
                
            rev_conn = gen_rev_conn(conn)
            
    sentence['words'] = {w['id']: w for w in words+new_words}
    sentence['conn'] = conn
    return sentence

def expand_lists(sentence):
    conn = sentence['conn']
    start_ids = conn[None]

    sentences_ids = set()
    for start_id in start_ids:
        sentences_ids.update(rec_gen_sentences(conn, [start_id]))

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
    sentence = expand_slash(sentence)
    sentence = expand_hyphen(sentence)

    return sentence


def query_index_db(kmer, db_conn):
    kmer_str = re.sub('\'', '\'\'', repr(kmer)) # handle quotes
    results = db_conn.execute(f"select st.onto_id, st.source, st.string, st.expanded_sentences from (select * from (select * from kmers where kmer='{kmer_str}') km left join kmer_to_string on km.id=kmer_to_string.kmer_id) ks inner join (select * from strings) st on ks.string_id=st.id")
    for onto_id, source, string, sentence in results:
        yield {'source': source, 'onto_id': onto_id, 'string': string, 'sentence': eval(sentence)[0]}

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

def all_word_query(sentence, matches, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    sentence_words = {v['word'].lower() for k,v in sentence['words'].items()}
    for match in matches:
        match_words = {v['word'].lower() for k,v in match['sentence']['words'].items() if v['word'].lower() not in stop_words}
        if len(match_words - sentence_words) == 0: yield match

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
            r = rec_conn_get_common_paths(sentence, match['sentence'], [(sentence_start_id,0)], [(match_start_id,0)], stop_words)
            paths.update({p[0] for p in r})

        good_paths = set()
        for path in paths:
            path_ids = {p[0] for p in path}

            # filter out paths that have too many gaps
            gap_sum = sum([p[1] for p in path])
            if gap_sum <= len(path_ids)/3: 
                good_paths.add(tuple(path_ids))

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



def query_sentence(sentence, db_conn):
    # do querying
    matches = kmer_query(sentence, query_index_db, [db_conn])            
    matches = all_word_query(sentence, matches)
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



def query_thesauruses(kmer, indexes):
    used_codes = set()
    for index in indexes:
        for n in range(1,4):
            if kmer in index[n].keys():
                for match in [{'code': k, 'sentence': v} for k,vs in index[n][kmer].items() for v in vs]:
                    if not match['code'] in used_codes:
                        used_codes.add(match['code'])
                        yield copy.deepcopy(match)

def query_names_indexes(onto_id, indexes):
    for index in indexes:
        try: 
            names = index[onto_id]
            for n in names: yield copy.deepcopy(n)
        except: pass

def expand_thesaurus(sentence, matches, query_f, f_args, stop_words={'of', 'type', 'with', 'and', 'the', 'or', 'due', 'in', 'to', 'by', 'as', 'a', 'an', 'is', 'for', '.', ',', ':', ';', '?', '-', '(', ')', '/', '\\', '\'', '"', '\n', '\t', '\r'}):
    match_names = set()
    for match in matches:
        # for the added word parents 
        match_path_ids = set()
        match_start_ids = set()
        match_end_ids = set()
        for p in match['paths']:
            match_path_ids.update(p)
            match_start_ids.add(p[0])
            match_end_ids.add(p[-1])

        match_start_ids = match_start_ids - {None}
        match_end_ids = match_end_ids - {None}

        match_path_words = {sentence['words'][i]['word'].lower() for i in match_path_ids if not sentence['words'][i]['word'].lower() in stop_words}
        
        names = query_f(match['code'], *f_args)
        
        # remove duplicates by standardising representation in hashed form
        for name in names:
            # check if the name adds any new words
            name_words = {w['word'].lower() for k,w in name['sentence']['words'].items() if not w['word'].lower() in stop_words}
            if len(name_words - match_path_words) > 0:
                name_sentence = {'words': {k:{'word': v['word'].lower(), 'id': v['id'], 'tag': v['tag'], 'type': v['type']} for k,v in name['sentence']['words'].items()}, 'conn': name['sentence']['conn']}
                match_names.add(repr({'sentence': name_sentence, 'match_path_ids': match_path_ids, 'match_start_ids': match_start_ids, 'match_end_ids': match_end_ids}))
        
    for name in match_names:
        name = eval(name)
        max_id = max(sentence['words'].keys())  # needs to be updated each time
        sentence['rev_conn'] = gen_rev_conn(sentence['conn'])  # needs to be updated each time
        
        name_to_sentence_map = {}
        
        # add words
        for k,v in name['sentence']['words'].items():
            i = v['id']
            new_id = max_id+i+1
            new_word = v.copy()
            new_word['id'] = new_id
            new_word['parent'] = name['match_path_ids']

            name_to_sentence_map[i] = new_id
            sentence['words'][new_id] = new_word

        # add conn from name
        name_rev_conn = gen_rev_conn(name['sentence']['conn'])
        name_start_ids = name['sentence']['conn'][None] - {None}
        name_end_ids = name_rev_conn[None] - {None}
        
        for k,vs in name['sentence']['conn'].items():
            if k is None:  # handle beginning of match
                for match_start_id in name['match_start_ids']:
                    for prev_match_start_id in sentence['rev_conn'][match_start_id]:
                        try: sentence['conn'][prev_match_start_id].update({name_to_sentence_map[v] for v in name_start_ids})
                        except: sentence['conn'][prev_match_start_id] = {name_to_sentence_map[v] for v in name_start_ids}
            else:
                new_k = name_to_sentence_map[k]
                for v in vs:
                    if v is None:  # handle end of match
                        for name_end_id in name_end_ids:
                            name_end_id = name_to_sentence_map[name_end_id]
                            for match_end_id in name['match_end_ids']:
                                try: sentence['conn'][name_end_id].update(sentence['conn'][match_end_id])
                                except: sentence['conn'][name_end_id] = set(sentence['conn'][match_end_id])
                    else:  # handle the other words
                        new_v = name_to_sentence_map[v]
                        try: sentence['conn'][new_k].add(new_v)
                        except: sentence['conn'][new_k] = {new_v}
                        
    return sentence

def expand_index(sentence, kmer_indexes, names_indexes):
    # find matches from the thesaurus indexes
    matches = kmer_query(sentence, query_thesauruses, [kmer_indexes], stop_words=set())
    matches = all_word_query(sentence, matches, stop_words=set())
    matches = loc_query(sentence, matches, stop_words=set())
    
    # substitute names from the matches into the sentence
    expanded_sentence = expand_thesaurus(sentence, matches, query_names_indexes, [names_indexes])
    
    return expanded_sentence
