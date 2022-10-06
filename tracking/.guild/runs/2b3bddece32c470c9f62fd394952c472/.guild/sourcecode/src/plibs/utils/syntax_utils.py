import spacy
from spacy.tokens import Span, Token
from spacy.tokens.doc import Doc 

def get_sdp_path(doc:Doc, head:int, tail:int, lca_matrix):
    """Get the Shortest-Dependency-Path between
    the specified head annd tail positions of the
    given doc 

    Args:
        doc (_type_): _description_
        head (_type_): index of the initial token
        tail (_type_): index of the end token
        lca_matrix (_type_): lca_matrix

    Returns:
        list: list of tokens that are the sdp from head to tail
    """
    lca = lca_matrix[head, tail]
  
    # collect head path
    current_node = doc[head]
    head_path = [current_node]
    if lca != -1: 
        if lca != head: 
            while current_node.head.i != lca:
                current_node = current_node.head
                head_path.append(current_node)
            head_path.append(current_node.head)
    
    # collect tail path
    current_node = doc[tail]
    tail_path = [current_node]
    if lca != -1: 
        if lca != tail: 
            while current_node.head.i != lca:
                current_node = current_node.head
                tail_path.append(current_node)
        tail_path.append(current_node.head)
    
    return head_path + tail_path[::-1][1:]

def get_closest(doc:Doc, anchor_token:Token, target_filter=(lambda t: t.ent_type_=='MED'), lca_matrix=None):
    """Get the closest target token to the anchor_token (within the dependency tree) 
    The target token has to pass the specified target_filter

    E.g. from the token 't' Find the closest token labeled as QTY :
        closest_qty, sdp = get_closest(doc, t, target_filter=(lambda t: t.ent_type_=='QTY'))

    E.g. from the token 't' Find the closest VERB token  :
        closest_qty, sdp = get_closest(doc, t, target_filter=(lambda t: t.pos_=='VERB'))

    Args:
        doc (Doc): _description_
        anchor_token (Token): _description_
        target_filter (tuple, optional): _description_. Defaults to (lambda t: t.ent_type_=='PER').
        lca_matrix (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple(target, best_sdp): found target token, best SDP from anchor_token to target token 
    """

    if not lca_matrix:
        lca_matrix = doc.get_lca_matrix()
    
    best_sdp = None
    target = None
    for t2 in doc:
        if target_filter(t2):
            sdp = get_sdp_path(doc, anchor_token.i, t2.i, lca_matrix)
            if not best_sdp or len(sdp) < len(best_sdp):
                best_sdp = sdp
                target = t2

    return target, best_sdp    


def get_ents_with_token(doc:Doc, t:Token, tgt_ent_types:list=['PER']):
    """Get the entities that contains the token 't'

    Args:
        doc (Doc): _description_
        t (Token): _description_
        tgt_ent_types (list, optional): entity types to be considered. Defaults to ['PER'].

    Returns:
        list: list of entities
    """

    rs = []
    for es in doc.ents:
        if not tgt_ent_types or es.label_ in tgt_ent_types:
            if es.start <= t.i and t.i < es.end:
                rs.append( es )
    return rs
  