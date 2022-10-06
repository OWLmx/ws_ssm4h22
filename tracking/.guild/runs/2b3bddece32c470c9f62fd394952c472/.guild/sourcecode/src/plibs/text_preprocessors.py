

from pydoc import Doc
import pandas as pd
from pandas import DataFrame
import spacy

# from negspacy.negation import Negex
# import negspacy
# from negspacy.termsets import termset
# ts = termset("en_clinical")
# print(ts.get_patterns())

# from tqdm import tqdm

class TextPreprocessor():
    # Base class so new strategies could implement it

    def __init__(self, **kwargs) -> None:
        pass

    def process(self, txt:str, ann:DataFrame, **kwargs) -> DataFrame:
        # segments text in spans (ideally associated with provided annotations)
        # generates a DF with text span per row and the associated file and 
        # Term(T) id to which the text is associated
        pass


class TextPreprocessorSpacySentences(TextPreprocessor):
    # Implementation based on default spacy's segmentation 

    def __init__(self, **kwargs) -> None:
        self.nlp = spacy.load("en_core_web_sm")

    def process(self, txt:str, ann:DataFrame, **kwargs) -> DataFrame:
        # segments text in spans (ideally associated with provided annotations)
        # generates a DF with text span per row and the associated file and 
        # Term(T) id to which the text is associated

        doc = self.nlp(txt)

        cr = '\n'
        rs = []
        txt_len = len(txt)
        br_total = txt.count('\n')
        br_count = 0
        prev_off_ini = 0
        for i,e in ann[ann.etype=='T'].iterrows():
            try:
                br_count += txt[prev_off_ini:e.off_ini].count('\n')
                char_span = doc.char_span(e.off_ini, e.off_end)
                rs.append({'file': e.file, 'id': e.id, 'term': char_span.text, 'text': char_span.sent.text.replace(cr, ' '), 
                'ith_pos': int((e.off_ini/txt_len)*100), # relative position based on characters
                'i10th_pos': int((e.off_ini/txt_len)*10), # relative position based on characters (tenths )
                'i5th_pos': int((e.off_ini/txt_len)*5), # relative position based on characters (fifths)
                'nth_pos': int((br_count/br_total)*100)  # relative position based on newlines
                })
                prev_off_ini = e.off_ini
            except Exception as err:
                print(f"Error segmenting {e.file}::{e.id} [{e.off_ini}:{e.off_end}] ", err)

        return pd.DataFrame(rs)




from spacy.tokens import Span 
from negspacy.negation import Negex
from quantulum3 import parser as quantumparser

''
class TextPreprocessorSpacySentencesPlusFeats(TextPreprocessor):
    # Implementation based on default spacy's segmentation 
    # extra features:
    #   - relative position within the note (standarized -> between 0 - 1.0)
    #   - action (verb) associated to the term 
    #   - detected quantities
    #   - negexs associated to term and to action

    def __init__(self, spacy_model="en_core_web_sm", using_char_span=True, **kwargs) -> None:
        
        self.nlp = spacy.load(spacy_model, disable = ['ner'])
        self.ruler = self.nlp.add_pipe("entity_ruler")
        self.nlp.add_pipe("negex")        
        # self.nlp.add_pipe("negex", config={"ent_types":["MED"]})

        self.using_char_span = using_char_span

    def _get_closest_pos(self, token, doc, pos='VERB', depth=0):

        depth+=1
        if depth>100: #safety break
            return None

        if token.pos_.startswith(pos):
            return token
        else:
            head = token.head
            if not head or head == token:
                return None 
            else:
                return self._get_closest_pos(head, doc, pos, depth)
                
    def _reannotate(self, txt, terms):
        try:
            # indentify already given NER (probably right)
            self.ruler.clear()    
            patterns = [{"label": "MED", "pattern": t} for t in terms]
            self.ruler.add_patterns(patterns)

            # apply pipeline with negex
            doc = self.nlp(txt)

            # get closest actions to MEDs
            try: 
                for t in doc:
                    if t.ent_type_ == 'MED':
                        verb = self._get_closest_pos(t, doc, pos='VERB')
                        if verb:
                            try:
                                doc.ents += tuple( [Span(doc, verb.i, verb.i+1, label= "ACT")] )
                                # doc.ents += tuple( [Span(doc, verb.i, verb.i+1, label= "ACTN" if verb._.negex else "ACT")] )
                            except:
                                pass
            except Exception as err:
                print(err)
                print(doc.ents)
                pass

            # apply quantity annotation
            try:
                quants = quantumparser.parse(txt)        
                anns = list(filter(None, [doc.char_span(q.span[0], q.span[1], label="QTY")  for q in quants]))
                if anns:
                    doc.ents = (doc.ents if doc.ents else tuple()) + tuple(anns)
            except:
                pass            

            #return negex results for all terms (probably always use only one)
            rs = [e._.negex for e in doc.ents]
            return doc, rs if rs else [False]
        except Exception as err:
            print(f"Error --->{terms} || {txt}")
            raise err
            # return None, [False]
        
    
    def _enrich(self, df):
        rs = [None]*len(df)
        rs_lemma = [None]*len(df)
        rs_negex = [False]*len(df)
        rs_nverbs = [0]*len(df)
        rs_nnouns = [0]*len(df)
        rs_qtys = [None]*len(df)
        # for i,e in tqdm(enumerate(df.itertuples()), total=len(df)):
        for i,e in enumerate(df.itertuples()):
            doc, negs = self._reannotate(e.text, [e.term])

            rs_negex[i] = any(negs)
            # get closest VERB to the term (considering only 1 term per sentence)
            if doc:
                term = None
                for t in doc:
                    if t.pos_.startswith('VERB'):
                        rs_nverbs[i] += 1
                    if t.pos_.startswith('NOUN'):
                        rs_nnouns[i] += 1
                    if t.ent_type_ == 'MED':
                        term = t
                        verb = self._get_closest_pos(t, doc, pos='VERB')
                        # rs.append(verb.text if verb and len(verb)>0 else None)
                        rs[i]= (verb.text if verb and len(verb)>0 else None)
                        rs_lemma[i]= (verb.lemma_ if verb and len(verb)>0 else None)
                        break

                #  select closest quantity to term------------
                if term:
                    tini = term.i+1
                    tend = term.i+1
                    # find candidate slice where to look for ents (no other MEDs nor VERB, NOUN, PROPN in between)  
                    for t in doc[tini:]:
                        if  t.ent_type_ == 'MED' or (t.ent_type_ != 'QTY' and  t.pos_ in ['VERB', 'NOUN', 'PROPN']  ):
                            break
                        tend = t.i
                    # collect matching ents within the slice 
                    qtys = [e.text for e in doc[tini:tend+1].ents if e.label_ =='QTY']
                    rs_qtys[i] = ' '.join(qtys) if qtys else None
                # / --------

        df['verb'] = rs
        df['verb_lemma'] = rs_lemma
        df['negex'] = rs_negex
        df['nnouns'] = rs_nnouns
        df['nverbs'] = rs_nverbs
        df['quants'] = rs_qtys

        df['nwords'] = df.text.map(lambda v: len(v.split(' ')))

        return df

    def get_container_sentence_(self, doc:Doc, off_ini:int, off_end:int, using_char_span=True):
        # get the container sentence of the passed offsents

        rel_off_ini = off_ini
        rel_off_end = off_end
        container = None
        if using_char_span:
            char_span = doc.char_span(off_ini, off_end)
            container = char_span.sent
        else: # look through all sentences
            for s in doc.sents:
                if (s.start_char <= off_ini <= s.end_char) and (s.start_char <= off_end <= s.end_char):
                    container = s
                    break

        if container:
            rel_off_ini = rel_off_ini - container.start_char
            rel_off_end = rel_off_end - container.start_char                    

        return container, doc.text[off_ini:off_end], (rel_off_ini, rel_off_end)

    def process(self, txt:str, ann:DataFrame, **kwargs) -> DataFrame:
        # segments text in spans (ideally associated with provided annotations)
        # generates a DF with text span per row and the associated file and 
        # Term(T) id to which the text is associated

        # for segmenting
        doc = self.nlp(txt)

        cr = '\n'
        rs = []
        txt_len = len(txt)
        prev_off_ini = 0
        # for i,e in tqdm(ann[ann.etype=='T'].iterrows(), total=len(ann[ann.etype=='T'])):
        for i,e in ann[ann.etype=='T'].iterrows():
            try:
                # char_span = doc.char_span(e.off_ini, e.off_end)
                # rs.append({'file': e.file, 'id': e.id, 'term': char_span.text, 'text': char_span.sent.text.replace(cr, ' '), 
                container_span, marked_text, relative_offsets = self.get_container_sentence_(doc, e.off_ini, e.off_end, using_char_span=self.using_char_span)
                rs.append({'file': e.file, 'id': e.id, 'term': marked_text, 'text': container_span.text.replace(cr, ' '), 
                'ith_pos': int((e.off_ini/txt_len)*100)/100, # relative position (01-100 -> 0.1 -> 1.0)
                'i10th_pos': int((e.off_ini/txt_len)*10)/10, # relative position based on characters (tenths )
                'i5th_pos': int((e.off_ini/txt_len)*5)/5, # relative position based on characters (fifths)
                'off_ini': relative_offsets[0],
                'off_end': relative_offsets[1],
                })
                prev_off_ini = e.off_ini
            except Exception as err:
                print(f"Error segmenting {e.file}::{e.id} [{e.off_ini}:{e.off_end}] ", err)

        rs = pd.DataFrame(rs)
        rs = self._enrich(rs)

        return rs


# di ct to access preprocessors by strategyName
strategies = {
    'SpacySentences': TextPreprocessorSpacySentences,
    'SpacySentencesPlusFeats': TextPreprocessorSpacySentencesPlusFeats
    }