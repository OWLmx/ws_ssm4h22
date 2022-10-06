"""Process brat file (ann) to build a dictionary and sequences of text and labels 
    to prepare for token classification

 Possible approach
 - replace terms with Tx (so multiword-terms become a single token)
 - segment sections into list of texts  (LoTe)
 - split texts into tokens -> list of tokens (LoTo)
 - generate List of Tags (LoTa) with ) values
 - loop sync LoTo & LoTa : replace in LoTa with actual T value from ann 
 - replace Tx for actual terms LoTo -> LoTo
 - split multi-word-terms in LoTo & LoTa -> iterate and if multiword split and duplicate correspondingly in LoTa -> possibly nested list 
 - flattize lists

    Returns:
        _type_: _description_
"""

from os import path, listdir
from functools import reduce
import operator
import re
import numpy as np
from tqdm import tqdm

from spacy.tokenizer import Tokenizer
import spacy
import json

cr ='\n'


def load_file(fdir, fname):
    ann, txt = [], []
    with open(path.join(fdir, f"{fname}.ann"), 'r' ) as f:
        ann = f.readlines()
    with open(path.join(fdir, f"{fname}.txt"), 'r' ) as f:
        txt = f.readlines()

    return ann, txt

def load_files(fpath_ann, fpath_txt):
    ann, txt = [], []
    with open(fpath_ann, 'r' ) as f:
        ann = f.readlines()
    with open(fpath_txt, 'r' ) as f:
        txt = f.readlines()

    return ann, txt


def get_Ts(anns, offset_sorted=True, reverse=True):
    # returns list like [['T36', 'Disposition', '8991', '8997', 'nexium'], ...    
    rs = list(filter(lambda a: a.startswith('T'), anns))
    rs = list(map(lambda e: reduce(operator.concat, [v.split(' ') if i==1 else [v] for i,v in enumerate(e.replace(cr, '').split('\t'))]) , rs ))
    if offset_sorted:
        rs = sorted(rs, key=lambda a: int(a[2]), reverse=reverse)
    return rs

def replace_term_with_placeholder(anns:list, txts:list, anns_ents=None):
    # probably better from backwards to kept valid offsets
    txt = ''.join(txts)
    if anns_ents is None:
        anns_ents = get_Ts(anns, offset_sorted=True, reverse=True)
    for t in anns_ents:
        try:
            txt = txt[:int(t[2])] + " _"  + t[0] + "_ " + txt[int(t[3]):] # add span around to ensure they are identified as independent during tokenization
        except:
            pass

    return txt


def segment_sections(txt, common_cr_clamped_to=None):
    # very basic approach for segmenting sections based on the "common" number of CRs

    txt = re.sub('\n[^\S\r\n]+\n', '\n\n', txt) # remove spaces between CRs
    crs = re.findall('(?:\r\n?|\n)+', txt) # cr-n-grams     
    crs = list(map(lambda c: len(c), crs)) # remove spaces to efectively differentiate cr-n-grams

    # compute median and max number of consecutive crs
    cr_median = int(np.median(crs) if crs else 1) # the most common n-cr-median (e.g. 2-cr ) so probably it is used as CR and not as section separator
    if common_cr_clamped_to:
        cr_median = min(cr_median, common_cr_clamped_to)
    cr_max = int(np.max(crs))
    
    # split with cr-n-grams above the median (reverse -> first longer cr markers)
    rs = [txt]
    for n in reversed(range(cr_median+1, cr_max+1 )):
        crx = cr*n
        rsi = []
        for txti in rs:
            rsi.extend(txti.split(crx))
        rs = rsi

    return rs

def ann_to_placeholder_dict(anns):
    # build fat dict for each placeholder 
    placeholders = get_Ts(anns, offset_sorted=True, reverse=True) 
    # process Ts, build dict of placeholders
    rs_dic = { f"_{t[0]}_": 
        {'term': t[4], 'offset_ini': int(t[2]), 'offset_end': int(t[3]), 'value': t[1] ,
            'events': {}, 'attrs': {}
        }   for t in placeholders if t[2].isnumeric() and t[3].isnumeric() } # [['T36', 'Disposition', '8991', '8997', 'nexium']...

    # process events
    ev2t = {} # event to T mapping
    for an in anns: # e.g. => 'E26\tNoDisposition:T26 \n'
        if an.startswith('E'):
            try:
                id, _ = an.strip().split('\t')
                val, tgt = _.split(':')
                rs_dic[f"_{tgt}_"]['events'][id] = {'value': val, 'attrs': {}}
                ev2t[id] = f"_{tgt}_"
            except:
                pass

    # process attributes
    att2t = {}
    for an in anns: # e.g. => 'A22\tActor E36 Physician\n'
        if an.startswith('A'):
            try:
                id, _ = an.strip().split('\t')
                att, tgt, val = _.split(' ')
                rs_dic[ ev2t[tgt] ]['events'][tgt]['attrs'][id] = {'name': att, 'value': val}
                att2t[id] = f"{ev2t[tgt]}.{tgt}"
            except:
                pass            
    
    rs_dic = {'placeholders': rs_dic, 'event_to_placeholder': ev2t, 'attr_to_placeholder': att2t}

    return rs_dic, placeholders
    





class BratProcessor():

    def __init__(self, attrs2hydrate=['Certainty', 'Actor', 'Action', 'Temporality', 'Negation']) -> None:

        # nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
        # with nlp.select_pipes(enable="parser"):
        # nlp = spacy.load("en_core_web_sm", enable=['tokenizer'])
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
        tokenizer = nlp.tokenizer

        # Very basic and conservative tokenize (so tranformer's specialized vocab can be recognized)
        special_cases = {"prn.": [{"ORTH": "prn."}]} #,"[0-9]{1,4}-[0-9]{1,4}-[0-9]{1,4}": []}
        prefix_re = re.compile(r'''^[\[\("']''')
        suffix_re = re.compile(r'''[\]\)\."']$''')
        infix_re = re.compile(r'''[-~]''')
        simple_url_re = re.compile(r'''^https?://''')

        def custom_tokenizer(nlp):
            return Tokenizer(nlp.vocab, rules=special_cases,
                                        prefix_search=prefix_re.search,
                                        suffix_search=suffix_re.search,
                                        infix_finditer=infix_re.finditer,
                                        url_match=simple_url_re.match)

        tokenizer = custom_tokenizer(nlp)
        tokenizer.token_match = re.compile("^_T[0-9]{1,3}_$").match # ensure Tx placeholders are not splitted
        self.tokenizer = tokenizer
        self.attrs2hydrate = attrs2hydrate




    def process_brat_for_tokencls(self, anns, txts):
        # def process_brat_for_tokencls(self, fdir, fpath):
        # anns, txts = load_file(fdir, fpath )

        # vanilla approach for hydrating, do not consider multi values of the same type for the same entity

        rs = {
            'text': [],
            'tags': {
                'entity': [],
                'event': []
            }
        }
        # add attributes to hydrate
        for a in self.attrs2hydrate:
            rs['tags'][f"attr_{a}"] = []
        
        ts_dict, ts = ann_to_placeholder_dict(anns)

        # build 
        ts_dict['text'] = []
        for s in segment_sections(replace_term_with_placeholder ( anns, txts, anns_ents=ts), common_cr_clamped_to=2): 
            t = s
            t = re.sub('[^\S\r\n]+', ' ', t) # reduce to single space
            loto = list(self.tokenizer(t)) # list of tokens
            lota = ['O']*len(loto) #  list of tags Os by default
            rs['text'].append( loto )

        # hydrate tags
        rs['tids'] = []
        for txt in rs['text']:
            tmp = {k:['O']*len(txt) for k in rs['tags'].keys()}
            tid = ['O']*len(txt)
            for i, tok in enumerate(txt):
                if tok.text in ts_dict['placeholders'].keys(): # is a placeholder
                    entity = ts_dict['placeholders'][tok.text]
                    tid[i] = tok.text # sequence with placeholders positioned
                    txt[i] = entity['term'] # getback original term into the tokenized sequence
                    for k in tmp:
                        if k=='entity':
                            tmp[k][i] = entity['value']
                        elif k=='event':
                            tmp[k][i] = entity['events'][list(entity['events'].keys())[0]]['value']
                            pass
                        elif k.startswith('attr_'):
                            attr_name = k[len('attr_'):]
                            for evk, ev in entity['events'].items():
                                for atrrk, attr in ev['attrs'].items():
                                    if attr['name'] == attr_name:
                                        tmp[k][i] = attr['value']
                            pass
                else:
                    txt[i] = tok.text # convert Token to text
                    pass

            ## append each built sequence of labels to their corresponding tag            
            rs['tids'].append(tid)
            for k in tmp:
                rs['tags'][k].append(tmp[k])
            
        # add hydrated tags
        ts_dict.update(rs)

        return ts_dict


    def process_brats_for_tokencls(self, ann_dir, txt_dir, output_dir):

        for f_ann in tqdm(listdir(ann_dir)):
            if f_ann.endswith('ann'):
                fname = path.splitext(f_ann)[0]
                anns, txts = load_files(fpath_ann=path.join(ann_dir, f_ann), fpath_txt=path.join(txt_dir, f"{fname}.txt"))
                rs = self.process_brat_for_tokencls(anns, txts)

                with open( path.join(output_dir, f"{fname}.json"), "w") as write_file:
                    json.dump(rs, write_file, indent=4)
        pass