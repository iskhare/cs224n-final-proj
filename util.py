
def split_bidirectional(puzzle):
    """
    split a directional puzzle (including "<" and ">" in test sentences)
    into two left-to-right unidirectional puzzles. (without ">" and "<")
    
    params:
        puzzle: directional puzzle (dict)
    returns:
        (ltr,rtl) tuple of unidirectional puzzles
        ltr:
            unidirectional puzzle (dict) 
            containing all sentences that go left-to-right in the original puzzle
        rtl:
            unidirectional puzzle (dict) 
            containing all sentences that go right-to-left in the original puzzle
            note that the source_language and target_language of rtl are swapped when compared to the input puzzle
    """
    ltr={}
    rtl={}
    same_keys=["id","task","url_ex","url_sol"]
    for k in same_keys:
        if k in puzzle:
            ltr[k]=puzzle[k]
            rtl[k]=puzzle[k]
    ltr["source_language"]=puzzle["source_language"]
    ltr["target_language"]=puzzle["target_language"]
    rtl["source_language"]=puzzle["target_language"]
    rtl["target_language"]=puzzle["source_language"]
    rtl["train"]=[]
    ltr["train"]=[]
    rtl["test"]=[]
    ltr["test"]=[]
    for sent in puzzle["train"]:
        rtl["train"].append(list(reversed(sent)))
        ltr["train"].append(sent)
    for sent in puzzle["test"]:
        if sent[2]==">":
            ltr["test"].append([sent[0],sent[1]])
        elif sent[2]=="<":
            rtl["test"].append([sent[1],sent[0]])
    if len(rtl["test"])==0:
        rtl=None
    if len(ltr["test"])==0:
        ltr=None
    return ltr,rtl
    
def merge_bidirectional(ltr,rtl):
    """
    merge two left-to-right unidirectional puzzles. (without ">" and "<")
    into a directional puzzle (including "<" and ">" in test sentences)
    
    NOTE: splitting and merging might change the order of the test sentences.
    after a merge, the test sentences contain all left-to-right sentences, 
    followed by all right-to-left sentences.
    
    params:
        ltr:
            unidirectional puzzle (dict) 
            containing all sentences that go left-to-right in the original puzzle
        rtl:
            unidirectional puzzle (dict) 
            containing all sentences that go right-to-left in the original puzzle
            note that the source_language and target_language of rtl must be swapped when compared to ltr
    returns:
        puzzle: 
            directional puzzle (dict) 
            containing test sentences from both puzzles
            will use source_language and target_language from ltr puzzle
    """
    have_ltr=(ltr is not None)
    have_rtl=(rtl is not None)
    if not (have_ltr or have_rtl):
        raise ValueError("Need to provide either left-to-right or right-to-left puzzle, got None for both.")
    merged_puzzle={}
    same_keys=["id","task","url_ex","url_sol"]
    
    for k in same_keys:
        if have_ltr and have_rtl and k in ltr and k in rtl:
            assert(ltr[k]==rtl[k])
        if have_ltr and k in ltr:
            merged_puzzle[k]=ltr[k]
        if have_rtl and k in rtl:
            merged_puzzle[k]=rtl[k]
    
    if have_ltr and have_rtl:
        assert(ltr["target_language"]==rtl["source_language"])
        assert(ltr["source_language"]==rtl["target_language"])
        
    if have_ltr:
        merged_puzzle["source_language"]=ltr["source_language"]
        merged_puzzle["target_language"]=ltr["target_language"]
        merged_puzzle["train"]=ltr["train"]
    else:
        merged_puzzle["source_language"]=rtl["target_language"]
        merged_puzzle["target_language"]=rtl["source_language"]
        merged_puzzle["train"]=[sent[::-1] for sent in rtl["train"]]
    
    merged_puzzle["test"]=[]
    if have_ltr:
        for sent in ltr["test"]:
            merged_puzzle["test"].append([sent[0],sent[1],">"])
    if have_rtl:
        for sent in rtl["test"]:
            merged_puzzle["test"].append([sent[1],sent[0],"<"])
    return merged_puzzle
    
def swap_bidirectional(puzzle):
    """
    swaps left and right languages of a directional puzzle, 
    while keeping the language direction of the test sentences intact.
    (e.g. 
    English -> Turkish 
    to 
    Turkish <- English)
    
    params:
        puzzle: directional puzzle (dict)
    returns:
        directional puzzle (dict) with languages swapped
    """
    return merge_bidirectional(*split_bidirectional(puzzle)[::-1])
    
def is_directional(puzzle):
    """
    check whether a file has directional information
    
    Raises an exception if some, but not all sentences have directional information
    
    params:
        puzzle:dict
    returns:
        boolean: True if puzzle has directional information,
                 False otherwise
    """
    num_dir=0
    num_uni=0
    for sent in puzzle["test"]:
        if len(sent)==3:
            num_dir+=1
        elif len(sent)==2:
            num_uni+=1
        else:
            raise Exception("FOUND TEST SENTENCE WITH ONLY 1 ENTRY!")
    assert (num_dir==0 or num_uni==0),"MIXED USE OF DIRECTIONAL AND UNDIRECTIONAL SENTENCES IN ONE PUZZLE"
    return (num_dir>0)

def add_dict_to_dict(dict1, dict2):
    """
    Add dict 1 to dict 2
    :param dict1:
    :param dict2:
    :return:
    """
    for k in dict1:
        dict2[k] += dict1[k]