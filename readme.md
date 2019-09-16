# Text Query Functions


## Parse XML

#### `text_filter(s: string)`

**returns:** `string`

**Description:** normalises whitespace

#### `split_tag_sentences(s: string, nlp: spacy parser)`

**returns:** `list` of `tuple`

**Description:** Splits string into list of POS-tagged words. Used in `handle_sentance(s, stop_words)`

#### `rec_parse(node: dict)`

**returns:** `dict`

**Description:** Recursive function that parses XML while retaining the nested structure and XML node tags. XML node contents are stored as split/tagged strings.

#### `handle_sentance(s: string, stop_words: set)`

**returns:** `dict`

**Description:** Takes XML node contents, tokenises, assigns POS tags and lables stop-words. Returns `string` node containing list of parsed sentences. Used in `rec_join_str(node)`.


## Pre-process the Parsed Sentences

#### `rec_join_str(node: dict)`

**returns:** `dict`

**Description:** Joins together sentences that are split by other tags (e.g. text formatting)

#### `generate_kmers(l: list, n: int)`

**yields:** `tuple`

**Description:** Splits list into kmers of length `n`, returns a sorted tuple

#### `generate_ordered_kmers(l: list, n: int)`

**yields:** `list`

**Description:** Splits list into kmers of length `n`, retains the original order (unlike `generate_kmers(l, n)`)

#### `gen_rev_conn(conn: dict)`

**returns:** `dict`

**Description:** Returns the inverse of the connectivity `dict`, `conn`. This `dict` is used to traverse a connected-sentence backwards.

#### `next_conn_skip_stop_words(sentence: dict, start_id: int, stop_words: set)`

**returns:** `set`

**Description:** Returns all word IDs that are next from the `start_id` in the connected sentence. It skips over stop words defined by `stop_words`.

#### `next_rev_conn_skip_stop_words(sentence: dict, rev_conn: dict, start_id: int, stop_words: set)`

**returns:** `set`

**Description:** Reverse of `next_conn_skip_stop_words(sentence, start_id, stop_words)` returning word IDs previous to `start_id` in the connected sentence. This uses `rev_conn` (the inverse of `conn`).

#### `rec_conn_gen_kmers(sentence: dict, kmer: list, n: int, stop_words: set)`

**returns:** `set`

**Description:** Starting at a single word ID in a sentence, generate all kmers of length `n` in a connected sentence, excluding stop words. This uses `next_conn_skip_stop_words(sentence, start_id, stop_words)` and `next_rev_conn_skip_stop_words(sentence, rev_conn, start_id, stop_words)`. 

#### `conn_gen_kmers(sentence: dict, n: int, stop_words: set)`

**returns:** `set`

**Description:** Using `rec_conn_gen_kmers(sentence, kmer, n, stop_words)` this generate all kmers of length `n` in a connected sentence, excluding stop words. Uses  but covers all of the words in the sentence.

#### `rec_gen_sentences(conn: dict, sentence: dict)`

**returns:** `set`

**Description:** Generate all possible combination of word IDs in a sentence given a connectivity map `conn`. This takes a connected sentence and return a list of all possible "normal" sentences.

#### `expand_brackets(sentence: dict)`

**returns:** `dict`

**Description:** Adds mappings to a sentence's `conn` showing that parentheses characters can be skipped, and parentheses contents can be ommitted.

#### `expand_slash(sentence: dict)`

**returns:** `dict`

**Description:** Adds mappings to a sentence's `conn` and words to the list possible words in the sentence to show that when words are seperated by a slash then any of those words can fill that position in the sentence.

#### `expand_hyphen(sentence: dict)`

**returns:** `dict`

**Description:** Adds mappings to a sentence's `conn` and words to the list possible words in the sentence to show that when words are seperated by a hyphen then that word can be split in two and the hyphen removed.

#### `expand_lists(sentence: dict)`

**returns:** `dict`

**Description:** Detects lists in the sentence using rules based on POS tags. Splits these lists into the **head**, **middle** (optional) and **tail** of the list, adds connections between words in these sections (all words of **head** connect to each item of **middle** which connect to all words of **tail**).


## Query the pre-processed sentences with an index

#### `rec_kmer_query(node: dict, loc: list, index: dict)`

**returns:** `dict`

**Description:** Recursively traverse parsed indications section `dict` keeping track of location `loc` in this nested structure. When the node is a `string` node then expand the sentences (parentheses, slash, hyphen, lists). For each sentence generate kmers and look these up in the kmer index (`index`). For each sentence with matches: store the location of this sentence and the list of matches (the match code and match string).

#### `all_words_query(sentences: dict)`

**returns:** `dict`

**Description:** For each match from `rec_kmer_query(node, loc, index)` check if all the words of the match are present in the sentence.

#### `rec_matching_word_paths(word_ids: set, word_next_ids: dict, word_prev_ids: dict, path: path)`

**returns:** `list`

**Description:** Recursively search out all the paths through the sentence between the words in `word_ids` while skipping stop words. Records the single gaps that are needed in the path. This uses `next_conn_skip_stop_words` and `next_rev_conn_skip_stop_words`. This is used in `loc_based_query(sentences)` to help determine the location of each match in the sentence and assess whether the words occur in sequence.

#### `loc_based_query(sentences: dict)`

**returns:** `dict`

**Description:** Determine the location of each match in the sentence and assess whether the words occur in sequence. All words of the match must occur in a sequence where one gap is allowed for every three words. Add these locs to the sentence match. This is used to remove some false positives and also highlight the match in a UI for a user to view, and in machine learning to build an automated system to find matches and seperate trua and false positives. 

#### `mesh_query(sections: list)`

**returns:** `list`

**Description:** For each indications section for a DailyMed SPL, find potential matches using `rec_kmer_query` and then remove false positives with `all_words_query` and `loc_based_query`. 


## Perform thesaurus expansions of terms

#### `conn_sentence_kmer_query(sentence: dict, indexes: list)`

**yields:** `dict`

**Description:** Queries a connected sentence (ie a sentence with a `word_index` and a `conn` dict) with a kmer. Kmers are generated using `conn_gen_kmers`, these are then queried against the indexes in the function argument.

#### `conn_sentence_all_word_query(sentence, matches)`

**yields:** `dict`

**Description:** Takes matches from `conn_sentence_kmer_query` and checks if all of the words of the match is present in the matching sentence. Only matches that pass this check are yielded.

#### `conn_sentence_loc_query(sentence: dict, matches: list)`

**yields:** `dict`

**Description:** Takes matches from `conn_sentence_all_word_query`, locates the matching words in the sentence and performs checks to see if the match is valid. Uses `conn_sentence_match_paths` to locate the matching words. A match location is valid if there are gaps with a frequency of 1 gap per 3 words or less, and the location contains all of the match words.

#### `conn_sentence_match_paths(sentence: dict, match_sentence: dict, stop_words: set)` 

**yields:** `dict`

**Description:** Used with `conn_sentence_loc_query`, finds paths through a connected sentence that connect the match words. Filters out paths that do not contain all the match words.

#### `expand_index(sentence: dict, indexes: list)`

**returns:** `list`

**Description:** joins `conn_sentence_kmer_query`, `conn_sentence_all_word_query` and `conn_sentence_loc_query` together to find all thesaurus matches in a sentence. Used for thesaurus expansion. 

#### `expand_thesaurus(matches: list, sentence:dict, indexes: list)`

**returns:** `dict`

**Description:** Takes validated thesaurus matches from `conn_sentence_loc_query` and performs thesaurus expansions on those matching words. The alternative words are added to `word_index`, and the beginning and end words of each match path are connected to these new words also. 

